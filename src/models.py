import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, EdgeGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss
import onnxruntime as ort


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')
        self.onnx_gen_weights_path = os.path.join(config.PATH, name + '_gen.onnx')

        self.generator = None
        self.discriminator = None
        self.onnx_session = None

    def load(self):
        print(f'Checking paths:\n ONNX path: {self.onnx_gen_weights_path}\n PyTorch path: {self.gen_weights_path}')
        if os.path.exists(self.onnx_gen_weights_path):
            print(f'Loading ONNX {self.name} generator...')
            self.onnx_session = ort.InferenceSession(self.onnx_gen_weights_path)
            print('ONNX model loaded successfully.')
        elif os.path.exists(self.gen_weights_path):
            print(f'Loading {self.name} generator...')
            data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']
            print('PyTorch model loaded successfully.')
        
        # Load discriminator only when training and if using pth
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print(f'Loading {self.name} discriminator...')
            data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)
            self.discriminator.load_state_dict(data['discriminator'])
            print('Discriminator loaded successfully.')

    def save(self):
        if self.generator is not None:
            print(f'\nSaving {self.name} generator as .pth...\n')
            torch.save({
                'iteration': self.iteration,
                'generator': self.generator.state_dict()
            }, self.gen_weights_path)
        
        if self.discriminator is not None:
            print(f'\nSaving {self.name} discriminator...\n')
            torch.save({
                'discriminator': self.discriminator.state_dict()
            }, self.dis_weights_path)

    def onnx_inference(self, input_tensor):
        if self.onnx_session is None:
            raise RuntimeError("ONNX model is not loaded.")
        
        print(f'Input tensor shape before reshaping: {input_tensor.shape}')
        
        # Ensure input_tensor is 4D: [batch_size, channels, height, width]
        if input_tensor.shape[1] == 5:  # Specific case where 5 channels are present
            # Here you need to determine how to handle the extra channel
            # Option 1: If the extra channel is not needed, you can remove it:
            input_tensor = input_tensor[:, :4, :, :]  # Keep only the first 4 channels
            
            # Option 2: Merge some channels if applicable (e.g., sum or average):
            # input_tensor = input_tensor[:, :3, :, :] + input_tensor[:, 3:, :, :]  # Example operation

        print(f'Input tensor shape after reshaping: {input_tensor.shape}')
        
        ort_inputs = {self.onnx_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = self.onnx_session.run(None, ort_inputs)
        
        output_tensor = torch.tensor(ort_outs[0]).to(input_tensor.device)
        print(f'Output tensor shape: {output_tensor.shape}')
        
        return output_tensor



class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # Load the generator
        if not os.path.exists(self.onnx_gen_weights_path):
            generator = EdgeGenerator(use_spectral_norm=True)
            if len(config.GPU) > 1:
                generator = nn.DataParallel(generator, config.GPU)
            self.generator = generator
            self.add_module('generator', self.generator)
        
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            discriminator = nn.DataParallel(discriminator, config.GPU)
        self.discriminator = discriminator
        self.add_module('discriminator', self.discriminator)

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        # Only initialize the optimizer if a PyTorch model is used
        if self.generator is not None:
            self.gen_optimizer = optim.Adam(
                params=self.generator.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def inference(self, input_tensor):
        if self.onnx_session:
            output = self.onnx_inference(input_tensor)
        else:
            output = self.generator(input_tensor)
        return output

    def process(self, images, edges, masks):
        self.iteration += 1


        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, *inputs):
        if self.onnx_session:
            # Prepare input for ONNX
            input_tensor = torch.cat(inputs, dim=1)
            output = self.onnx_inference(input_tensor)
        else:
            output = self.generator(torch.cat(inputs, dim=1))
        return output

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()

class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # Load the generator
        if not os.path.exists(self.onnx_gen_weights_path):
            generator = InpaintGenerator()  # Removed 'use_spectral_norm' argument
            if len(config.GPU) > 1:
                generator = nn.DataParallel(generator, config.GPU)
            self.generator = generator
            self.add_module('generator', self.generator)

        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            discriminator = nn.DataParallel(discriminator, config.GPU)
        self.discriminator = discriminator
        self.add_module('discriminator', self.discriminator)

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)

        # Only initialize the optimizer if a PyTorch model is used
        if self.generator is not None:
            self.gen_optimizer = optim.Adam(
                params=self.generator.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def inference(self, input_tensor):
        if self.onnx_session:
            output = self.onnx_inference(input_tensor)
        else:
            output = self.generator(input_tensor)
        return output


    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss


        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, *inputs):
        if self.onnx_session:
            # Prepare input for ONNX
            input_tensor = torch.cat(inputs, dim=1)
            output = self.onnx_inference(input_tensor)
        else:
            output = self.generator(torch.cat(inputs, dim=1))
        return output


    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()
