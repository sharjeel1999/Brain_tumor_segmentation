from .unet2d import UNet
from .encoder_2d import Encoder_2d #UNet_2D
from .encoder_3d import Encoder_3d #UNet_3D
from .reg_discriminator import Discriminator
from .decoder import Decoder_2D #Decoder_2D
from .decoder_3d import Decoder_3D

from .encoder_2d_reduced import UNet_2D_red
from .encoder_3d_reduced import UNet_3D_red
from .decoder_3d_red import Decoder_3D_red
from .densenet_3d import generate_model
from .densenet_decoder import Dense_Decoder_2D
from .dense_encoders import Dense_encoder_2d, Dense_encoder_3d
from .conditional_random_fields import CRF
# from. VTUnet import SwinTransformerSys3D
from .decoder_mod import Decoder_2D_mod
from .complete_3D_unets import ResidualUNet3D
from .nnUnet import nnUNet
from .nnUNet_2d import nnUNet_2D