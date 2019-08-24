# -*- coding: utf-8
"""keras 实现卷积自编码网络。用于OCR图像编码"""
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model, load_model
from figs_load import figs_array


class ConvolutionAutoEncoder(object):
    """卷积自编码网络

    Attributes
    ----------
        input_shape: tuple, (width, height, channel)
            单个图片的张量尺寸

        model_dir：str, directory
            模型文件的存储路径

        cae_f：str, file name
            模型文件的完整路径与文件名

        cae_model：keras Model objective
            卷积自编码器，实现从x_in到x_out的映射

        encoder：keras Model objective
            编码器，实现从x_in到x_short的映射

    Methods
    -------
        model_construct：
            构建编码解码网络

        cae_compile：
            指定损失函数与优化方法，编译网络结构

        fit：
            训练模型

        reload：
            从文件中加载（恢复）模型

        cmp_show：
            可视化一张图片的解码效果，与原图片对比

        predict：
            编码一张图片
    """

    def __init__(self, input_shape, model_dir='./cae_model', optimizer='adam',
                 loss='binary_crossentropy'):
        self.input_shape = input_shape
        self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        cae_file = os.path.join(self.model_dir, 'model_cae.h5')
        encoder_file = os.path.join(self.model_dir, 'model_enc.h5')
        self.cae_f = cae_file
        self.encoder_f = encoder_file
        # 模型初始化
        self.cae_model = None
        self.encoder = None
        self.model_construct()
        self.cae_compile(optimizer=optimizer, loss=loss)
        return

    def model_construct(self):
        """构建网络"""
        input_img = Input(shape=self.input_shape)
        # encoder
        x = input_img
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        encoded = x
        # decoder
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        decoded = x

        encoder = Model(input_img, encoded)
        en_decoder = Model(input_img, decoded)
        self.encoder = encoder
        self.cae_model = en_decoder

    def cae_compile(self, optimizer, loss):
        self.cae_model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

    def fit(self, xs, **args):
        fit_args = {'epochs': 10, 'batch_size': 64, 'shuffle': True,
                    'verbose': 1}
        fit_args.update(args)
        self.cae_model.fit(xs, xs, **fit_args)
        # save
        self.cae_model.save(self.cae_f)
        self.encoder.save(self.encoder_f)

    def reload(self):
        if os.path.isfile(self.cae_f):
            self.cae_model = load_model(self.cae_f)
        else:
            raise ValueError
        if os.path.isfile(self.encoder_f):
            self.encoder = load_model(self.encoder_f)
        else:
            raise ValueError

    def cmp_show(self, fig):
        if self.cae_model is None:
            self.reload()
        fig_pred = self.cae_model.predict(fig[None, :])
        plt.subplot(1, 2, 1)
        plt.imshow(fig[:, :, 0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(fig_pred[0, :, :, 0], cmap='gray')
        plt.show()

    def predict(self, xs):
        if self.encoder is None:
            self.reload()
        return self.encoder.predict(xs)


if __name__ == "__main__":
    Xs, Ys = figs_array()
    in_shape = Xs.shape[1:]
    print("input shape:", in_shape)
    cae = ConvolutionAutoEncoder(in_shape)
    cae.fit(Xs)
    # print(cae.predict(Xs[:3]).shape)
    for fig in Xs:
        cae.cmp_show(fig)
    pass
