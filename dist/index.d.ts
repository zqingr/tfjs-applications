import { Model } from '@tensorflow/tfjs'

/**
 * ResNet Version 1 Model builder [a]
 *
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
 *
 * @param input_shape (number[]): shape of input image tensor
 * @param depth (number): number of core convolutional layers
 * @param numClasses (number): number of classes (CIFAR10 has 10)
 * @returns model (Model): tfjs model instance
 */
export type resnetV1 = ({ inputShape, depth, numClasses }: {inputShape: number[], depth: number, numClasses?: number}) => Model

/**
 * ResNet Version 2 Model builder [b]
 *
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
 *
 * @param input_shape (number[]): shape of input image tensor
 * @param depth (number): number of core convolutional layers
 * @param numClasses (number): number of classes (CIFAR10 has 10)
 * @returns model (Model): tfjs model instance
 */
export type resnetV2 = ({ inputShape, depth, numClasses }: {inputShape: number[], depth: number, numClasses?: number}) => Model