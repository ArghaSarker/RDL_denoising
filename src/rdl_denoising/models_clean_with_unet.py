import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as F
import tensorflow as tf
import tifffile
from csbdeep.utils import normalize
from sim_fitting import cal_modamp, get_otf
from skimage.transform import resize
from tensorflow.keras import callbacks, regularizers
from tensorflow.keras.layers import (
    AveragePooling2D,
    Conv2D,
    Input,
    Lambda,
    LayerNormalization,
    LeakyReLU,
    UpSampling2D,
    add,
    multiply,
)
from tensorflow.keras.models import Model
from tqdm import tqdm

# def print_memory_usage():
#     process = psutil.Process(os.getpid())  # Get the current Python process
#     memory_info = process.memory_info()   # Get memory usage details for this process
#     print()
#     print('----------------------------->>       Process Memory Usage       <<-----------------------------')
#     print(f"RSS (Resident Set Size): {memory_info.rss / (1024 ** 3):.2f} GB")
#     print(f"VMS (Virtual Memory Size): {memory_info.vms / (1024 ** 3):.2f} GB")
#     print(f"Shared Memory: {memory_info.shared / (1024 ** 3):.2f} GB" if memory_info.shared else "N/A")
#     print(f"Data Memory: {memory_info.data / (1024 ** 3):.2f} GB")
#     print(f"Text Memory: {memory_info.text / (1024 ** 3):.2f} GB")
#     print()


def CALayer2D(input, input_height, input_width, channel, reduction=16):
    W = AveragePooling2D(pool_size=(input_height, input_width))(input)
    W = Conv2D(
        channel // reduction,
        kernel_size=1,
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(W)
    W = LayerNormalization()(W)  # added layer here
    W = Conv2D(
        channel,
        kernel_size=1,
        activation="sigmoid",
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(W)
    W = UpSampling2D(size=(input_height, input_width))(W)
    mul = multiply([input, W])
    return mul


def RCAB2D(input, input_height, input_width, channel):
    conv = Conv2D(
        channel,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(input)
    conv = LeakyReLU(alpha=0.1)(conv)
    conv = LayerNormalization()(conv)  # added layer here
    conv = Conv2D(
        channel,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(conv)
    conv = LeakyReLU(alpha=0.1)(conv)
    conv = LayerNormalization()(conv)  # added layer here
    att = CALayer2D(conv, input_height, input_width, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup2D(input, input_height, input_width, channel):
    conv = input
    n_RCAB = 5
    for _ in range(n_RCAB):
        conv = RCAB2D(conv, input_height, input_width, channel)
    output = add([conv, input])
    return output


def Denoiser(input_shape, n_rg=(2, 5, 5)):

    inputs1 = Input(input_shape)
    inputs2 = Input(input_shape)
    # --------------------------------------------------------------------------------
    #                      extract features of generated image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(
        64,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(inputs1)
    conv = LeakyReLU(alpha=0.1)(conv0)
    conv = LayerNormalization()(conv)  # added layer here
    for _ in range(n_rg[0]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(
        64,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(conv)
    conv1 = LeakyReLU(alpha=0.1)(conv)
    conv1 = LayerNormalization(name="mfe_out")(conv1)  # added layer here

    # Name the PFE output
    # pfe_out = Lambda(lambda x: x, name="pfe_out")(conv1)
    # --------------------------------------------------------------------------------
    #                      extract features of noisy image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(
        64,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(inputs2)
    conv = LeakyReLU(alpha=0.1)(conv0)
    conv = LayerNormalization()(conv)  # added layer here
    for _ in range(n_rg[1]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(
        64,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(conv)
    conv2 = LeakyReLU(alpha=0.1)(conv)
    conv2 = LayerNormalization(name="pfe_out")(
        conv2
    )  # added layer here with a name for extraction
    # Name the MFE output
    # mfe_out = Lambda(lambda x: x, name="mfe_out")(conv2)

    # --------------------------------------------------------------------------------
    #                              merge features
    # --------------------------------------------------------------------------------
    weight1 = Lambda(lambda x: x * 1)
    weight2 = Lambda(lambda x: x * 1)
    conv1 = weight1(conv1)
    conv2 = weight2(conv2)

    conct = add([conv1, conv2])
    conct = Conv2D(
        64,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(conct)
    conct = LeakyReLU(alpha=0.1)(conct)
    conct = LayerNormalization()(conct)  # added layer here
    conv = conct

    for _ in range(n_rg[2]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conct])

    conv = Conv2D(
        256,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(conv)
    conv = LeakyReLU(alpha=0.1)(conv)
    conv = LayerNormalization()(conv)  # added layer here

    CA = CALayer2D(conv, input_shape[0], input_shape[1], 256, reduction=16)
    conv = Conv2D(
        input_shape[2],
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1.0e-4),
    )(CA)

    output = LeakyReLU(alpha=0.1, name="final_out")(conv)

    model = Model(
        inputs=[inputs1, inputs2], outputs=output, name="RDL_denoiser"
    )
    return model


class Train_RDL_Denoising(tf.keras.Model):
    def __init__(
        self,
        srmodel,
        denmodel,
        loss_fn,
        optimizer,
        parameters,
        PSF="given",
        verbose=False,
    ):
        super().__init__()
        self.srmodel = srmodel
        self.denmodel = denmodel
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.PSF = PSF
        self.verbose = verbose
        self.parameters = parameters
        self.epochs = self.parameters["epochs"]
        self.nphases = self.parameters["nphases"]
        self.ndirs = self.parameters["ndirs"]
        self.space = self.parameters["space"]
        self.Ny = self.parameters["Ny"]
        self.Nx = self.parameters["Nx"]
        self.phase_space = 2 * np.pi / self.nphases
        self.scale = self.parameters["scale"]
        self.dxy = self.parameters["dxy"]
        self.sigma_x = self.parameters["sigma_x"]
        self.sigma_y = self.parameters["sigma_y"]
        self.dxy = self.parameters["dxy"]
        self.sr_model_dir = self.parameters["sr_model_dir"]
        self.den_model_dir = self.parameters["den_model_dir"]
        self.log_dir = self.parameters["log_dir"]
        self.batch_size = self.parameters["batch_size"]
        [self.Nx_hr, self.Ny_hr] = [self.Nx * self.scale, self.Ny * self.scale]
        [self.dx_hr, self.dy_hr] = [
            x / self.scale for x in [self.dxy, self.dxy]
        ]

        xx = self.dx_hr * np.arange(-self.Nx_hr / 2, self.Nx_hr / 2, 1)
        yy = self.dy_hr * np.arange(-self.Ny_hr / 2, self.Ny_hr / 2, 1)
        [self.X, self.Y] = np.meshgrid(xx, yy)

        self.dkx = 1.0 / (self.Nx * self.dxy)
        self.dky = 1.0 / (self.Ny * self.dxy)
        self.prol_OTF = None
        self.otf_path = self.parameters["otf_path"]
        self.results_path = self.parameters["results_path"]
        self.dkr = np.min([self.dkx, self.dky])

        if self.PSF == "given":  # read out the PSF from the RDL_Sim pakage

            self.OTF, self.prol_OTF, self.PSF = get_otf(
                self.otf_path,
                self.Nx_hr,
                self.Ny_hr,
                self.dkx,
                self.dky,
                self.dkr,
            )
            print("Information OTF how its read")
            # (256, 256) float64 1.0 -0.0006461802372663593
            print(
                f"{self.OTF.shape} {self.OTF.dtype} {np.max(self.OTF)} {np.min(self.OTF)}"
            )
            print()
            print("Information PSF how its read")
            # (256, 256) float64 0.011698316186498404 1.0611680661324662e-10
            print(
                f"{self.PSF.shape} {self.PSF.dtype} {np.max(self.PSF)} {np.min(self.PSF)}"
            )
            print()
            print("Information prol_OTF how its read")
            # (366,) float64 1.0 -0.0006461802372663593
            print(
                f"{self.prol_OTF.shape} {self.prol_OTF.dtype} {np.max(self.prol_OTF)} {np.min(self.prol_OTF)}"
            )

        else:
            self.PSF /= np.sum(self.PSF)
            self.OTF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.PSF)))
            self.OTF /= np.sum(self.OTF)

        if self.verbose:
            fig, axes = plt.subplots(1, 2, figsize=(15, 15))
            axes[0].imshow(self.PSF)
            axes[0].set_title("PSF")

            axes[1].imshow(abs(self.OTF))
            axes[1].set_title("OTF")

            plt.tight_layout()
            plt.show()
            plt.savefig(
                f"{self.results_path }/PSF_OTF.png", bbox_inches="tight"
            )

    def _phase_computation(self, img_SR, modamp, cur_k0_angle, cur_k0):

        phase_list = -np.angle(modamp)
        img_gen = []
        # gen_pattern = []
        # pattrned_img_fft_save = []
        for d in range(self.ndirs):
            alpha = cur_k0_angle[d]

            for i in range(self.nphases):
                kxL = cur_k0[d] * np.pi * np.cos(alpha)
                kyL = cur_k0[d] * np.pi * np.sin(alpha)
                kxR = -cur_k0[d] * np.pi * np.cos(alpha)
                kyR = -cur_k0[d] * np.pi * np.sin(alpha)
                phOffset = phase_list[d] + i * self.phase_space

                interBeam = np.exp(
                    1j * (kxL * self.X + kyL * self.Y + phOffset)
                ) + np.exp(1j * (kxR * self.X + kyR * self.Y))
                # interBeam = (np.exp(1j * (kxL * self.X + kyL * self.Y + phOffset)) + np.exp(1j * (kxR * self.X + kyR * self.Y)))

                pattern = normalize(np.square(np.abs(interBeam)))

                patterned_img_fft = (
                    F.fftshift(F.fft2(pattern * img_SR)) * self.OTF
                )

                modulated_img = np.abs(F.ifft2(F.ifftshift(patterned_img_fft)))

                modulated_img = normalize(
                    cv2.resize(modulated_img, (self.Ny, self.Nx))
                )

                img_gen.append(modulated_img)

        img_gen = np.asarray(img_gen)

        return img_gen

    def _get_cur_k(self, image_gt):

        cur_k0, modamp = cal_modamp(
            np.array(image_gt).astype(np.float32),
            self.prol_OTF,
            self.parameters,
        )

        cur_k0_angle = np.array(np.arctan2(cur_k0[:, 1], cur_k0[:, 0]))
        cur_k0_angle[1 : self.parameters["ndirs"]] = (
            cur_k0_angle[1 : self.parameters["ndirs"]] + np.pi
        )
        cur_k0_angle = -(cur_k0_angle - np.pi / 2)

        for nd in range(self.parameters["ndirs"]):
            if (
                np.abs(cur_k0_angle[nd] - self.parameters["k0angle_g"][nd])
                > 0.05
            ):
                cur_k0_angle[nd] = self.parameters["k0angle_g"][nd]
        cur_k0 = np.sqrt(np.sum(np.square(cur_k0), 1))
        given_k0 = 1 / self.parameters["space"]
        cur_k0[np.abs(cur_k0 - given_k0) > 0.1] = given_k0

        return cur_k0, cur_k0_angle, modamp

    def reshape_to_3_channels(self, batch):
        print()
        print("reshape_to_3_channels data received")
        print(f"batch shape: reshape_to_3_channels  {batch.shape}")

        B, H, W, C = batch.shape
        print(f"B: {B} , H: {H}, W: {W} , C: {C} ")
        assert C % self.ndirs == 0, "The last dimension must be divisible by 3"
        new_batch_size = B * (C // self.ndirs)
        print(
            f" new_batch_size : { new_batch_size} B: {B} , H: {H}, W: {W} , C: {C} "
        )
        return batch.reshape(new_batch_size, H, W, self.nphases)

    def reshape_to_9_channels(self, batch):
        B, H, W, C = batch.shape
        new_batch_size = int(B / (self.ndirs * self.nphases / C))
        return batch.reshape(new_batch_size, H, W, self.ndirs * self.nphases)

    def plot_batch_images(self, pattern_batch, filename="output.png"):
        """
        Plots and saves a figure of images in a batch where all images and channels are in a single row.
        Each set of 3 subplots corresponds to one image's channels (C1, C2, C3).

        Parameters:
        - pattern_batch: numpy array of shape (B, H, W, C), where B is the batch size, H is height, W is width, and C is the number of channels.
        - filename: string, the name of the file to save the figure (default is 'output.png').
        """

        num_images = pattern_batch.shape[0]
        num_channels = pattern_batch.shape[-1]
        total_subplots = num_images * num_channels

        fig, axs = plt.subplots(1, total_subplots, figsize=(55, 30))

        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for i in range(num_images):
            img = pattern_batch[i]
            for j in range(num_channels):
                channel_img = img[:, :, j]
                axs[i * num_channels + j].imshow(channel_img, cmap="gray")
                axs[i * num_channels + j].set_title(f"Img {i+1}, Ch {j+1}")
                axs[i * num_channels + j].axis("off")  # Hide axes

        plt.savefig("filename", bbox_inches="tight")

        plt.show()

    def fit(self, data, data_val):
        x, y = data
        x_val, y_val = data_val

        tensorboard_callback = callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1
        )

        lrate = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,  # here is teh change
            patience=4,
            mode="auto",
            min_delta=1e-4,
            cooldown=0,
            min_lr=max(1e-4 * 0.1, 1e-5),
            verbose=1,
        )
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(
            str(self.den_model_dir),
            monitor="val_loss",
            save_best_only=True,  # here is teh changeF
            save_weights_only=True,
            mode="auto",
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",  # Also monitor validation loss
            patience=6,  # Number of epochs with no improvement after which to stop
            mode="min",  # We want to minimize validation loss
            restore_best_weights=True,  # Restore model weights from the epoch with the best validation loss
        )
        # with tf.device('/GPU:0'):
        print()
        print()
        print(
            "inside the model  SR block, this is what going in for model prediction"
        )
        print(f"x shape: {x.shape} min: {np.min(x)} max: {np.max(x)}")

        # here is teh change, you need it if DFCAN data was True. If False, Not needed
        # why, the DFCAn data expects Channel first, and PU-net training expects C first.
        # if you do not have to do tranpose, you are saving meory and training time.
        if x.shape[-1] == 9:
            x_sr = tf.transpose(x, perm=(0, 3, 1, 2))
        else:
            x_sr = x

        print()
        print()
        print(f"the shpe   of x_sr : {x_sr.shape}")

        sr_y_predict = [
            self.srmodel.predict(x_s[..., np.newaxis].numpy(), axes="ZYXC")
            for x_s in x_sr
        ]
        # sr_y_predict = self.srmodel(x, training=False) # here is teh change

        sr_y_predict = normalize(np.array(sr_y_predict))

        print()
        print(f"this came out of sr prediction : {sr_y_predict.shape}")

        # this is what came out of SR preciction
        # sr_y_predict shape: (20, 256, 256, 1) min: 0.03381483256816864 max: 0.6741549372673035

        print()
        print("this is what came out of SR preciction")
        print(
            f"sr_y_predict shape: {sr_y_predict.shape} min: {np.min(sr_y_predict)} max: {np.max(sr_y_predict)}"
        )
        print()
        sr_y_predict = tf.squeeze(sr_y_predict, axis=-1)  # Batch, Ny, Nx, 1

        # we have to normalize the SR data
        # sr_y_predict = normalize(sr_y_predict)

        print(f"sr predcitec inside train {sr_y_predict.shape}")

        list_image_gen = []
        list_image_in = []
        list_image_gt = []

        batch_size_for_data_pre_process = 300
        num_batches = int(
            np.ceil(x.shape[0] / batch_size_for_data_pre_process)
        )
        # print()
        # print()
        # print("before the bacth preprocessing startes on the training data")
        # print_memory_usage()

        # Process data in batches with a progress bar
        for batch_idx in tqdm(
            range(num_batches), desc="Processing batches for training data"
        ):
            # Calculate the start and end indices for the current batch
            start_idx = batch_idx * batch_size_for_data_pre_process
            end_idx = min(
                (batch_idx + 1) * batch_size_for_data_pre_process, x.shape[0]
            )

            # Extract the current batch from x, y, and sr_y_predict
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            sr_y_predict_batch = sr_y_predict[start_idx:end_idx]

            # print()
            # print(
            #     "before each mini batch startes for training data pre processing"
            # )
            # print_memory_usage()
            # print()

            for i in range(x_batch.shape[0]):
                # for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                img_in = x_batch[i : i + 1][0]
                # image_in_ori = x[i:i+1][0]
                # list_image_input_ori.append(image_in_ori)
                img_SR = sr_y_predict_batch[i : i + 1][0]
                # print(f'this is teh shape of img sr inside loop: {img_SR.shape}')
                image_gt = y_batch[i : i + 1][0]
                # image_gt_ori = y[i:i+1][0]
                # print(f'inside fit, tis is what I am ffeeding inside gte_vur_k {image_gt.shape}')
                cur_k0, cur_k0_angle, modamp = self._get_cur_k(
                    image_gt=image_gt
                )

                image_gen = self._phase_computation(
                    img_SR, modamp, cur_k0_angle, cur_k0
                )

                image_gen = np.transpose(image_gen, (1, 2, 0))
                # reversing the image gen channels.
                image_gen = image_gen[:, :, ::-1]

                list_image_gen.append(
                    image_gen
                )  # here is list image gen # ----> being append to the list outside of bathing for loop

                # Intensity equalization for the input image and ground truth image

                # img_in_t = np.transpose(img_in, (2, 0, 1)).copy()
                # gt_t = np.transpose(image_gt, (2, 0, 1)).copy()

                # # Intensity equalization for the input image

                # mean_th_in = np.mean( img_in_t[self.nphases, :, :])
                # for d in range(1, self.ndirs):
                #     data_d =  img_in_t[d * self.nphases:(d + 1) * self.nphases, :, :]
                #     img_in_t[d * self.nphases:(d + 1) * self.nphases, :, :] = data_d * mean_th_in / np.mean(data_d)

                # mean_th_gt = np.mean(gt_t[self.nphases, :, :])
                # for d in range(self.ndirs):
                #     data_d = gt_t[d * self.nphases:(d + 1) * self.nphases, :, :]
                #     gt_t[d * self.nphases:(d + 1) * self.nphases, :, :] = data_d * mean_th_gt / np.mean(data_d)

                # # Transpose back to the original shape (128,128,9)
                # img_in = np.transpose(img_in_t, (1, 2, 0))
                # image_gt = np.transpose(gt_t, (1, 2, 0))

                # list_image_gen.append(image_gen)
                list_image_in.append(
                    img_in
                )  # ----> being append to the list outside of bathing for loop
                list_image_gt.append(
                    image_gt
                )  # ----> being append to the list outside of bathing for loop
            # print()
            # print(
            #     "after each batch pre processing training data pre processing ---> Memory requirement"
            # )
            # print_memory_usage()
            # print()

        input_MPE_batch = np.asarray(list_image_gen)
        input_PFE_batch = np.asarray(list_image_in)
        gt_batch = np.asarray(list_image_gt)

        print()
        print()

        print("this are teh shapes that I want to work with")
        print(f"input_MPE_batch shape: {input_MPE_batch.shape} ")
        print(f"input_PFE_batch shape: {input_PFE_batch.shape} ")
        print(f"gt_batch shape: {gt_batch.shape} ")

        print()
        print()

        def reshape_channels(array):
            """
            Reshapes an array of shape (m, 128, 128, 9) to (m * 3, 128, 128, 3).

            Parameters:
            - array: numpy array of shape (m, 128, 128, 9)

            Returns:
            - reshaped array of shape (m * 3, 128, 128, 3)
            """
            # print(f'from reshape_channels:received:  {array.shape}')
            # Check if the input array has the correct number of channels (9)
            if array.shape[-1] != 9:
                raise ValueError("The last dimension must be 9 channels.")

            # Step 1: Reshape from (m, 128, 128, 9) to (m, 128, 128, 3, 3)
            if array.shape[-2] == 128:
                reshaped = array.reshape(array.shape[0], 128, 128, 3, 3)
                reshaped = reshaped.transpose(0, 3, 1, 2, 4).reshape(
                    -1, 128, 128, 3
                )
                # print(f'from reshape_channels: {reshaped.shape}')

                return reshaped
            else:
                reshaped = array.reshape(array.shape[0], 256, 256, 3, 3)
                reshaped = reshaped.transpose(0, 3, 1, 2, 4).reshape(
                    -1, 256, 256, 3
                )
                # print(f'from reshape_channels: {reshaped.shape}')

                return reshaped

        def restore_channels(array):
            """
            Restores an array of shape (m * 3, 128, 128, 3) back to (m, 128, 128, 9).

            Parameters:
            - array: numpy array of shape (m * 3, 128, 128, 3)

            Returns:
            - reshaped array of shape (m, 128, 128, 9)
            """
            # print(f'from restore_channels: received: {array.shape}')
            if array.shape[2] == 128:
                # Step 1: Reshape the array from (30, 128, 128, 3) to (10, 3, 128, 128, 3)
                reshaped = array.reshape(-1, 3, 128, 128, 3)

                # Step 2: Transpose the array back to the shape (10, 128, 128, 3, 3)
                reshaped = reshaped.transpose(0, 2, 3, 1, 4)

                # Step 3: Reshape to (10, 128, 128, 9) by collapsing the last two dimensions
                restored = reshaped.reshape(-1, 128, 128, 9)

                # print(f'from restore_channels: restored shape: {restored.shape}')
                return restored
            else:
                # Step 1: Reshape the array from (30, 128, 128, 3) to (10, 3, 128, 128, 3)
                reshaped = array.reshape(-1, 3, 256, 256, 3)

                # Step 2: Transpose the array back to the shape (10, 128, 128, 3, 3)
                reshaped = reshaped.transpose(0, 2, 3, 1, 4)

                # Step 3: Reshape to (10, 128, 128, 9) by collapsing the last two dimensions
                restored = reshaped.reshape(-1, 256, 256, 9)

                # print(f'from restore_channels: restored shape: {restored.shape}')
                return restored

            # Step 2: Transpose to swap the 3rd and 4th axes and reshape to (m * 3, 128, 128, 3)

        input_MPE_batch_new_reshape = reshape_channels(input_MPE_batch)
        input_PFE_batch_new_reshape = reshape_channels(input_PFE_batch)
        gt_batch_new_reshape = reshape_channels(gt_batch)

        input_MPE_batch_new_reshape = np.asarray(input_MPE_batch_new_reshape)
        input_PFE_batch_new_reshape = np.asarray(input_PFE_batch_new_reshape)
        gt_batch_new_reshape = np.asarray(gt_batch_new_reshape)

        #  Pre_process for Validation Data
        # input_height_val = x_val.shape[1]
        # input_width_val = x_val.shape[2]
        # channels_val = x_val.shape[-1]
        x_val, y_val = data_val
        # this part is for validation data
        # with tf.device('/GPU:0'):

        x_sr_val = tf.transpose(x_val, perm=(0, 3, 1, 2))  # here is teh change
        print()
        print()
        print(f"the shpe   of x_sr : {x_sr_val.shape}")

        sr_y_predict_val = [
            self.srmodel.predict(x_s[..., np.newaxis].numpy(), axes="ZYXC")
            for x_s in x_sr_val
        ]
        # sr_y_predict = self.srmodel(x, training=False) # here is teh change

        sr_y_predict_val = normalize(np.array(sr_y_predict_val))

        # sr_y_predict_val = self.srmodel.predict(x_val)
        sr_y_predict_val = tf.squeeze(
            sr_y_predict_val, axis=-1
        )  # Batch, Ny, Nx, 1

        # we have to normaliez the sr data
        # sr_y_predict_val = normalize(sr_y_predict_val)

        # list stays out of the loop
        list_image_gen_val = []
        list_image_in_val = []
        list_image_gt_val = []

        batch_size_for_data_pre_process_val = 300
        num_batches_val = int(
            np.ceil(x_val.shape[0] / batch_size_for_data_pre_process_val)
        )

        # Process data in batches with a progress bar
        for batch_idx in tqdm(
            range(num_batches_val),
            desc="Processing batches for validation data",
        ):
            # Calculate the start and end indices for the current batch
            start_idx = batch_idx * batch_size_for_data_pre_process_val
            end_idx = min(
                (batch_idx + 1) * batch_size_for_data_pre_process_val,
                x_val.shape[0],
            )

            # Extract the current batch from x, y, and sr_y_predict
            x_batch = x_val[start_idx:end_idx]
            y_batch = y_val[start_idx:end_idx]
            sr_y_predict_batch = sr_y_predict_val[start_idx:end_idx]

            # list_image_patterns_val = []

            for i in range(x_batch.shape[0]):
                img_in_val = x_batch[i : i + 1][0]
                img_SR_val = sr_y_predict_batch[i : i + 1][0]
                image_gt_val = y_batch[i : i + 1][0]
                cur_k0, cur_k0_angle, modamp = self._get_cur_k(
                    image_gt=image_gt_val
                )  # can be also image_in

                image_gen_val = self._phase_computation(
                    img_SR_val, modamp, cur_k0_angle, cur_k0
                )

                image_gen_val = np.transpose(image_gen_val, (1, 2, 0))
                # gen_pattern_val = np.transpose(gen_pattern_val, (1, 2, 0))

                image_gen_val = image_gen_val[:, :, ::-1]

                # intensoity equalization for the input image and ground truth image

                # img_in_t_val = np.transpose(img_in_val, (2, 0, 1)).copy()
                # gt_t_val = np.transpose(image_gt_val, (2, 0, 1)).copy()

                # # Intensity equalization for the input image

                # mean_th_in_val = np.mean( img_in_t_val[self.nphases, :, :])
                # for d in range(1, self.ndirs):
                #     data_d =  img_in_t_val[d * self.nphases:(d + 1) * self.nphases, :, :]
                #     img_in_t_val[d * self.nphases:(d + 1) * self.nphases, :, :] = data_d * mean_th_in_val / np.mean(data_d)

                # mean_th_gt_val = np.mean(gt_t_val[self.nphases, :, :])
                # for d in range(self.ndirs):
                #     data_d = gt_t[d * self.nphases:(d + 1) * self.nphases, :, :]
                #     gt_t_val[d * self.nphases:(d + 1) * self.nphases, :, :] = data_d * mean_th_gt_val / np.mean(data_d)

                # # Transpose back to the original shape (128,128,9)
                # img_in_val = np.transpose(img_in_t_val, (1, 2, 0))
                # image_gt_val = np.transpose(gt_t_val, (1, 2, 0))

                # mean_th_in = np.mean(img_in[: self.nphases, :, :])

                list_image_gen_val.append(image_gen_val)
                list_image_in_val.append(img_in_val)
                list_image_gt_val.append(image_gt_val)

        input_MPE_batch_val = np.asarray(list_image_gen_val)
        input_PFE_batch_val = np.asarray(list_image_in_val)
        gt_batch_val = np.asarray(list_image_gt_val)
        #  = np.asarray( list_image_patterns_val)

        input_MPE_batch_val_new_reshape = reshape_channels(input_MPE_batch_val)
        input_PFE_batch_val_new_reshape = reshape_channels(input_PFE_batch_val)
        gt_batch_val_new_reshape = reshape_channels(gt_batch_val)

        # x_val, y_val = data_val
        # val_data = [input_MPE_batch_val, input_PFE_batch_val] , gt_batch_val
        val_data = [
            input_MPE_batch_val_new_reshape,
            input_PFE_batch_val_new_reshape,
        ], gt_batch_val_new_reshape

        # if self.verbose:
        if False:
            pass

            # print(f'input MPE {input_MPE_batch.shape}, input PFE {input_PFE_batch.shape},gt {gt_batch.shape}, patterns {patterns_batch.shape} , patterns_fft {pattrned_img_fft_save_batch.shape}')
            # plot_batches(self, input_MPE_batch_new_reshape, input_PFE_batch_new_reshape, gt_batch_new_reshape, patterns_batch_new_reshape,
            #                    input_MPE_batch_restored,input_PFE_batch_restored, gt_batch_restored ,patterns_batch_restored, image_gt_ori, image_input_ori) # have to pass epoch

        print("model summary")
        # self.denmodel.summary()
        print("model outputs")
        print(self.denmodel.outputs)
        print("model layerss and names")
        for layer in self.denmodel.layers:
            if layer.name in ["pfe_out", "mfe_out"]:
                print(layer.name, "output shape:", layer.output_shape)

        # original_weights = self.denmodel.get_weights()

        # print()
        # print("before actual training starts,  the status of memory")
        # print_memory_usage()
        # print()
        # print()

        print("Number of GPUs used: 1")
        # self.denmodel.compile(loss=self.loss_fn, optimizer=self.optimizer)

        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        # self.denmodel.fit([input_MPE_batch_new_reshape, input_PFE_batch_new_reshape], gt_batch_new_reshape , validation_data = val_data, batch_size=self.batch_size,
        #                     epochs=self.epochs, shuffle=True, verbose = True,
        #                     callbacks=[lrate, hrate, srate, early_stopping, tensorboard_callback])
        # print('saving the  trained DN model')
        # print('model summary')
        #     new_denmodel = Denoiser((self.Ny, self.Nx, self.nphases))

        # # Compile the new model with the same loss and optimizer
        #     new_denmodel.compile(loss=self.loss_fn, optimizer=self.optimizer)

        #     # Load the original weights into the new model
        #     new_denmodel.set_weights(original_weights)
        print()
        print("this is being feed into actual model trainng")
        print(
            f"input_MPE_batch_new_reshape shape: {input_MPE_batch_new_reshape.shape}"
        )
        print(
            f"input_PFE_batch_new_reshape shape: {input_PFE_batch_new_reshape.shape}"
        )
        print(f"gt_batch_new_reshape shape: {gt_batch_new_reshape.shape}")
        print()
        print()
        print(f"val_data shape: {val_data[0][0].shape}")
        print(f"val_data shape: {val_data[0][1].shape}")
        print(f"val_data shape: {val_data[1].shape}")

        print()

        def plot_batches_only(input_MPE_batch, input_PFE_batch, gt_batch):
            num_batches = input_MPE_batch.shape[0]
            random_indices = np.random.choice(num_batches, 5, replace=False)

            fig, axes = plt.subplots(5, 3, figsize=(15, 15))

            for i, idx in enumerate(random_indices):
                axes[i, 0].imshow(input_MPE_batch[idx])
                axes[i, 0].set_title(f"input MPE batch {idx}")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(input_PFE_batch[idx])
                axes[i, 1].set_title(f"input PFE batch {idx}")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(gt_batch[idx])
                axes[i, 2].set_title(f"gt batch {idx}")
                axes[i, 2].axis("off")
            plt.tight_layout()
            plt.savefig(
                f"{self.results_path}/DN_input_features_from_branch.tiff",
                bbox_inches="tight",
            )
            plt.show()

        if self.verbose:
            plot_batches_only(
                input_MPE_batch_new_reshape,
                input_PFE_batch_new_reshape,
                gt_batch_new_reshape,
            )

        # print()
        # print("before model training starts,  the status of memory")
        # print_memory_usage()

        # print()
        print(
            f"input_MPE_batch_new_reshape : {input_MPE_batch_new_reshape.shape}"
        )
        print(
            f"input_PFE_batch_new_reshape : {input_PFE_batch_new_reshape.shape}"
        )
        print(f"gt_batch_new_reshape : {gt_batch_new_reshape.shape}")

        print()
        print(f"val_data MPE shape: {val_data[0][0].shape}")
        print(f"val_data PFE shape: {val_data[0][1].shape}")
        print(f"val_data GT shape: {val_data[1].shape}")

        # Train the new model on multiple GPUs
        self.denmodel.fit(
            [input_MPE_batch_new_reshape, input_PFE_batch_new_reshape],
            gt_batch_new_reshape,
            validation_data=val_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            verbose=True,
            callbacks=[
                lrate,
                hrate,
                srate,
                early_stopping,
                tensorboard_callback,
            ],
        )

        self.denmodel.save(self.den_model_dir)  # not saving for debugging
        print("model has been saved")

    def predict(self, data_x):

        print(
            "        #############     #####################     Prediction Started                     ############################                                 #############"
        )
        x = data_x
        # y = data_gt
        print(f"inside prediction data received {x.shape} ")
        input_height = x.shape[1]
        input_width = x.shape[2]
        # channels = x.shape[-1]
        # with tf.device('/GPU:0'):

        print(f"for prediction, received data shape: {x.shape}")

        x_sr_predict = tf.transpose(x, perm=(0, 3, 1, 2))  # here is teh change
        print()
        print()
        print(f"the shpe   of x_sr : {x_sr_predict.shape}")

        #    this is what u_net_prediction_needs
        #    (5, 9, 128, 128, 1)

        sr_y_predict = [
            self.srmodel.predict(x_s[..., np.newaxis].numpy(), axes="ZYXC")
            for x_s in x_sr_predict
        ]
        # sr_y_predict = self.srmodel(x, training=False) # here is teh change

        sr_y_predict = normalize(np.array(sr_y_predict))

        # sr_y_predict = self.srmodel.predict(x)
        sr_y_predict = tf.squeeze(sr_y_predict, axis=-1)  # Batch, Ny, Nx, 1
        print(
            f"this is what came out of SR preciction for full image {sr_y_predict.shape}"
        )
        # import tifffile as tiff
        # tiff.imwrite('/share/klab/argha/files_to_transfer/to_fetch_SR_u_nte-Full/sr_y_predict_inside_full_prediction.tif', sr_y_predict.numpy())
        if sr_y_predict.shape[1] in [1014, 1006]:
            print("I am resizing to 1004 x 1004")
            temp_image = tf.squeeze(sr_y_predict)
            temp_image_resize = resize(
                temp_image, (1004, 1004), anti_aliasing=True
            )
            sr_y_predict = temp_image_resize[np.newaxis, ...]

        if sr_y_predict.shape[1] in [604]:
            print("I am resizing to 600 x 600")
            temp_image = tf.squeeze(sr_y_predict)
            temp_image_resize = resize(
                temp_image, (600, 600), anti_aliasing=True
            )
            sr_y_predict = temp_image_resize[np.newaxis, ...]

        # tiff.imwrite('/share/klab/argha/files_to_transfer/to_fetch_SR_u_nte-Full/sr_y_predict_inside_full_prediction_after_resize.tif', sr_y_predict.numpy() if hasattr(sr_y_predict, 'numpy') else sr_y_predict)

        print(
            f"sr predcitec inside prediction after squeueze now {sr_y_predict.shape}"
        )

        # we have to normalize the SR data
        # sr_y_predict = normalize(sr_y_predict)

        # list_image_gen = []
        # list_image_in = []
        list_image_SR = []
        # list_prediction = []
        predictions_list = []

        batch_size_for_data_pre_process = 300
        num_batches = int(
            np.ceil(x.shape[0] / batch_size_for_data_pre_process)
        )

        # Process data in batches with a progress bar
        for batch_idx in tqdm(
            range(num_batches), desc="Processing batches for prediction data"
        ):
            # Calculate the start and end indices for the current batch
            start_idx = batch_idx * batch_size_for_data_pre_process
            end_idx = min(
                (batch_idx + 1) * batch_size_for_data_pre_process, x.shape[0]
            )

            # Extract the current batch from x, y, and sr_y_predict
            x_batch = x[start_idx:end_idx]
            # y_batch = y[start_idx:end_idx]

            sr_y_predict_batch = sr_y_predict[start_idx:end_idx]

            # list_image_gt = []
            for i in range(x_batch.shape[0]):
                img_in = x_batch[i : i + 1][0]  # --> 128,128,9
                # print(f'img_in before phase computation {img_in.shape} dtype: {img_in.dtype} max: {np.max(img_in)} min: {np.min(img_in)}')
                img_SR = sr_y_predict_batch[i : i + 1][0]
                # img_gt = y_batch[i:i+1][0]
                # print(f'img_SR before phase computation {img_SR.shape} dtype: {img_SR.dtype} max: {np.max(img_SR)} min: {np.min(img_SR)}')
                list_image_SR.append(img_SR)
                # image_gt = y[i:i+1][0]
                cur_k0, cur_k0_angle, modamp = self._get_cur_k(
                    image_gt=img_in
                )  # can be also image_in
                # print()
                # print('comming out of prediction branch')
                # print(f'cur_k0 shape: {cur_k0} cur_k0_angle shape: {cur_k0_angle} modamp shape: {modamp}')
                # print()
                # print(f'img_SR before phase computation {img_SR.shape}')
                # img_gen, gen_pattern, pattrned_img_fft_save
                # img_gen, gen_pattern, pattrned_img_fft_save = self._phase_computation(img_SR, modamp, cur_k0_angle, cur_k0)
                img_gen = self._phase_computation(
                    img_SR, modamp, cur_k0_angle, cur_k0
                )
                # print(f'image_gen from phase computation {image_gen.shape}')

                img_gen = np.transpose(img_gen, (1, 2, 0))
                img_gen = img_gen[:, :, ::-1]

                # intensity equalization for the input image and ground truth image

                # img_in = np.transpose(img_in, (2, 0, 1))
                # # print(f'img_in before intensity qualization {img_in.shape}')

                # mean_th_img_in = np.mean( img_in[self.nphases, :, :])
                # for d in range(1, self.ndirs):
                #     data_d =  img_in[d * self.nphases:(d + 1) * self.nphases, :, :]
                #     img_in[d * self.nphases:(d + 1) * self.nphases, :, :] = data_d * mean_th_img_in / np.mean(data_d)

                # img_in = np.transpose(img_in, (1, 2, 0))

                def plot_prediction_1_to_3(
                    prediction, filename="the_name.png"
                ):
                    """
                    Plots a (1, 128, 128, 3) array as three separate images (128, 128, 1), (128, 128, 2), (128, 128, 3).

                    Parameters:
                    - prediction: numpy array of shape (1, 128, 128, 3)

                    """
                    # Step 1: Remove the batch dimension to get shape (128, 128, 3)
                    # print(f'this is the orediction received : {prediction.shape}')
                    img = np.squeeze(prediction, axis=0)
                    # img_g = np.transpose(prediction, (1,2,0))
                    img_g = img
                    # print(f'after trabspose : {img_g.shape}')

                    # Step 2: Split the 3 channels
                    channel_1 = img_g[:, :, 0]
                    channel_2 = img_g[:, :, 1]
                    channel_3 = img_g[:, :, 2]

                    # Step 3: Plot the 3 channels in a single row
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                    axs[0].imshow(channel_1, cmap="gray")
                    axs[0].set_title("Channel 1")
                    axs[0].axis("off")

                    axs[1].imshow(channel_2, cmap="gray")
                    axs[1].set_title("Channel 2")
                    axs[1].axis("off")

                    axs[2].imshow(channel_3, cmap="gray")
                    axs[2].set_title("Channel 3")
                    axs[2].axis("off")

                    # Step 4: Save the figure
                    plt.tight_layout()
                    tifffile.imwrite(
                        f'{filename.split(".")[0]}.tiff', prediction
                    )
                    plt.savefig(filename, dpi=300)
                    plt.show()

                pred = []
                for d in range(self.ndirs):
                    Gen = img_gen[
                        :, :, d * self.ndirs : (d + 1) * self.nphases
                    ]
                    input_image = img_in[
                        :, :, d * self.ndirs : (d + 1) * self.nphases
                    ]

                    input1 = np.reshape(
                        Gen, (1, input_height, input_width, self.nphases)
                    )
                    input2 = np.reshape(
                        input_image,
                        (1, input_height, input_width, self.nphases),
                    )

                    pr = self.denmodel.predict([input1, input2], verbose=False)

                    pr = np.squeeze(pr[0])

                    # print(f'output of prediction each : {pr.shape}')
                    for pha in range(self.nphases):

                        pred.append(np.squeeze(pr[:, :, pha]))

                predictions_list.append(pred)

        predictions_list = np.asarray(predictions_list)
        # print(f'predictions_list : {predictions_list.shape}')
        predictions_list = np.transpose(predictions_list, (0, 2, 3, 1))

        image_SR_batch = np.asarray(list_image_SR)

        return predictions_list, image_SR_batch
