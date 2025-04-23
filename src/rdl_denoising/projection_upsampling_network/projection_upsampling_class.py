import warnings

import numpy as np
from csbdeep.internals import nets
from csbdeep.internals.train import DataWrapper
from csbdeep.models import CARE, ProjectionCARE, ProjectionConfig
from csbdeep.utils import (
    _raise,
    axes_check_and_normalize,
    axes_dict,
)
from csbdeep.utils.tf import BACKEND as K
from csbdeep.utils.tf import (
    IS_TF_1,
    CARETensorBoardImage,
    keras_import,
)
from keras import layers
from keras.regularizers import l2
from skimage.transform import resize

Model = keras_import("models", "Model")
Input, Conv3D, MaxPooling3D, UpSampling3D, UpSampling2D, Lambda, Multiply = (
    keras_import(
        "layers",
        "Input",
        "Conv3D",
        "MaxPooling3D",
        "UpSampling3D",
        "UpSampling2D",
        "Lambda",
        "Multiply",
    )
)
softmax = keras_import("activations", "softmax")


class ProjectionUpsamplingConfig(ProjectionConfig):

    def __init__(
        self,
        axes="ZYX",
        n_channel_in=1,
        n_channel_out=1,
        probabilistic=False,
        unet_n_depth=3,
        train_loss="mse",
        unet_n_first=48,
        unet_kern_size=3,
        train_epochs=400,
        train_batch_size=4,
        train_learning_rate=0.0001,
        upsampling_factor=2,
        train_steps_per_epoch=20,
        allow_new_parameters=True,
        **kwargs,
    ):

        self.axes = axes
        self.n_channel_in = n_channel_in
        self.n_channel_out = n_channel_out
        self.probabilistic = probabilistic
        self.unet_n_depth = unet_n_depth
        self.unet_n_first = unet_n_first
        self.unet_kern_size = unet_kern_size
        self.train_batch_size = train_batch_size
        self.train_epochs = train_epochs
        self.train_learning_rate = train_learning_rate
        self.train_loss = train_loss
        self.upsampling_factor = upsampling_factor
        self.train_steps_per_epoch = train_steps_per_epoch
        # self.train_reduce_lr = train_reduce_lr
        self.update_parameters(allow_new_parameters, **kwargs)
        super().__init__(
            axes=axes,
            n_channel_in=n_channel_in,
            n_channel_out=n_channel_out,
            probabilistic=probabilistic,
            unet_n_depth=unet_n_depth,
            train_epochs=self.train_epochs,
            train_batch_size=self.train_batch_size,
            unet_n_first=self.unet_n_first,
            train_loss=train_loss,
            unet_kern_size=self.unet_kern_size,
            train_learning_rate=self.train_learning_rate,
            # train_reduce_lr=self.train_reduce_lr,
            train_steps_per_epoch=self.train_steps_per_epoch,
            **kwargs,
            # train_steps_per_epoch = self.train_steps_per_epoch,
        )


class ProjectionUpsampling(ProjectionCARE):

    def _build(self):
        # get parameters
        proj = self.proj_params
        proj_axis = axes_dict(self.config.axes)[proj.axis]

        # define surface projection network (3D -> 2D)
        inp = u = Input(self.config.unet_input_shape)
        print(f"shape of input: {inp.shape}")

        def conv_layers(u):
            for i in range(proj.n_conv_per_depth):
                # Use a fixed kernel size of 9,3,3 for the first convolution layer only
                kernel_size = (9, 3, 3) if i == 0 else proj.kern
                u = Conv3D(
                    proj.n_filt,
                    kernel_size,
                    padding="same",
                    kernel_regularizer=l2(0.0001),
                )(u)
                u = layers.BatchNormalization()(u)
                u = layers.Activation("relu")(u)
            return u

        # down
        for _ in range(proj.n_depth):
            u = conv_layers(u)
            u = MaxPooling3D(proj.pool)(u)
        # middle
        u = conv_layers(u)
        # up
        for _ in range(proj.n_depth):
            u = UpSampling3D(proj.pool)(u)
            u = conv_layers(u)
        u = Conv3D(1, proj.kern, padding="same", activation="linear")(u)
        # convert learned features along Z to surface probabilities
        # (add 1 to proj_axis because of batch dimension in tensorflow)
        u = Lambda(lambda x: softmax(x, axis=1 + proj_axis))(u)  # noqa: E731
        # multiply Z probabilities with Z values in input stack
        u = Multiply()([inp, u])
        # perform surface projection by summing over weighted Z values
        u = Lambda(lambda x: K.sum(x, axis=1 + proj_axis))(u)  # noqa: E731
        # Perform upsampling
        upsampling_factor = self.config.upsampling_factor
        u = UpSampling2D(size=(upsampling_factor, upsampling_factor))(u)
        # for learning the upsampling from 128 to 256
        # u = Conv2DTranspose(
        #     filters=16,
        #     kernel_size=(3, 3),
        #     strides=(upsampling_factor, upsampling_factor),
        #     padding="same",
        #     kernel_regularizer=l2(0.0001),
        #     activation="linear",
        # )(u)
        # u = layers.BatchNormalization()(u)
        # u = layers.Activation("relu")(u)

        model_projection = Model(inp, u)

        # define denoising network (2D -> 2D)
        # (remove projected axis from input_shape)
        input_shape = list(self.config.unet_input_shape)
        del input_shape[proj_axis]
        model_denoising = nets.common_unet(
            n_dim=self.config.n_dim - 1,
            n_channel_out=self.config.n_channel_out,
            prob_out=self.config.probabilistic,
            residual=self.config.unet_residual,
            n_depth=self.config.unet_n_depth,
            kern_size=self.config.unet_kern_size,
            n_first=self.config.unet_n_first,
            last_activation=self.config.unet_last_activation,
        )(tuple(input_shape))

        # chain models together
        return Model(inp, model_denoising(model_projection(inp)))

    # extar added code
    #########################################################################################
    def train(self, X, Y, validation_data, **kwargs):
        proj_axis = self.proj_params.axis
        proj_axis = 1 + axes_dict(self.config.axes)[proj_axis]
        Y.shape[proj_axis] == 1 or _raise(ValueError())
        Y = np.take(Y, 0, axis=proj_axis)
        try:
            X_val, Y_val = validation_data
            # Y_val.shape[proj_axis] == 1 or _raise(ValueError())
            validation_data = X_val, np.take(Y_val, 0, axis=proj_axis)
        except:  # noqa: E722
            pass

        return ProjectionCARE.train(self, X, Y, validation_data, **kwargs)

    ###############################################################################################

    @property
    def _config_class(self):
        return ProjectionUpsamplingConfig


class AlbumentationsDataWrapper(DataWrapper):
    def __init__(
        self, X, Y, batch_size, length, augmenter=None, keras_kwargs=None
    ):
        # Pass None to base so we can override augmentation ourselves
        super().__init__(
            X, Y, batch_size, length, augmenter=None, keras_kwargs=keras_kwargs
        )
        self.augmenter = augmenter

    def __getitem__(self, i):

        idx = self.batch(i)
        X_batch, Y_batch = self.X[idx], self.Y[idx]
        if self.augmenter is not None:
            X_aug, Y_aug = [], []
            for x_img, y_img in zip(X_batch, Y_batch):

                x_img = np.array(x_img).squeeze().transpose(1, 2, 0)
                # print(f'now 2 : x_img : {x_img.shape} , y_img : {y_img.shape}  ')
                xi = resize(
                    x_img, (256, 256), preserve_range=True, anti_aliasing=True
                )

                yi = np.repeat(y_img, 9, axis=-1)

                augmented = self.augmenter(image=xi, mask=yi)
                x_augmented = augmented["image"]
                y_augmented = augmented["mask"]

                xi_aug = resize(
                    x_augmented,
                    (128, 128),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                yi_aug = y_augmented[:, :, 1]
                yi_aug = yi_aug[..., np.newaxis]

                xi_aug = xi_aug.transpose(2, 0, 1)
                xi_aug = xi_aug[..., np.newaxis]

                X_aug.append(xi_aug)
                Y_aug.append(yi_aug)
            X_batch = np.array(X_aug)
            Y_batch = np.array(Y_aug)

        return X_batch, Y_batch


class ProjectionCARE(CARE):
    def train(
        self,
        X,
        Y,
        validation_data,
        epochs=None,
        steps_per_epoch=None,
        augmenter=None,
    ):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        Y : :class:`numpy.ndarray`
            Array of target images.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        (
            (
                isinstance(validation_data, (list, tuple))
                and len(validation_data) == 2
            )
            or _raise(
                ValueError("validation_data must be a pair of numpy arrays")
            )
        )

        n_train, n_val = len(X), len(validation_data[0])
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn(
                "small number of validation images (only %.1f%% of all images)"
                % (100 * frac_val)
            )
        axes = axes_check_and_normalize("S" + self.config.axes, X.ndim)
        ax = axes_dict(axes)

        for a, div_by in zip(axes, self._axes_div_by(axes)):
            n = X.shape[ax[a]]
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axis %s"
                    " (which has incompatible size %d)" % (div_by, a, n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        if (
            self.config.train_tensorboard
            and self.basedir is not None
            and not IS_TF_1
            and not any(
                isinstance(cb, CARETensorBoardImage) for cb in self.callbacks
            )
        ):
            self.callbacks.append(
                CARETensorBoardImage(
                    model=self.keras_model,
                    data=validation_data,
                    log_dir=str(self.logdir / "logs" / "images"),
                    n_images=3,
                    prob_out=self.config.probabilistic,
                )
            )

        training_data = AlbumentationsDataWrapper(
            X,
            Y,
            self.config.train_batch_size,
            length=epochs * steps_per_epoch,
            augmenter=augmenter,
        )

        fit = (
            self.keras_model.fit_generator if IS_TF_1 else self.keras_model.fit
        )
        history = fit(
            iter(training_data),
            validation_data=validation_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.callbacks,
            verbose=1,
        )
        self._training_finished()

        return history
