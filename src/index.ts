import { resnetV2 } from './resnet'
import * as tf from '@tensorflow/tfjs'
import { Cifar10 } from './datasets/cifar10'

// Load the binding:
require('@tensorflow/tfjs-node') // Use '@tensorflow/tfjs-node-gpu' if running with GPU.

const model = resnetV2({ inputShape: [32, 32, 3], depth: 11 })

const optimizer = tf.train.adam()
model.compile({
  optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
})
// model.summary()

async function train (data: Cifar10) {
  // The entire dataset doesn't fit into memory so we call train repeatedly
  // with batches using the fit() method.
  console.log(1)
  const { xs: x, ys: y } = data.nextTrainBatch()
  console.log(2)
  const history = await model.fit(
    x.reshape([50000, 32, 32, 3]) as any, y as any, {
      batchSize: 32,
      epochs: 200,
      callbacks: {
        onBatchEnd: (epoch: number, log: tf.Logs) => {
          console.log(epoch, log)
          return tf.nextFrame()
        },
        onEpochBegin: (epoch: number, log: tf.Logs) => {
          console.time('Epoch training time')
          console.groupCollapsed('epoch times:', epoch)
          return tf.nextFrame()
        },
        onEpochEnd: (epoch: number, log: tf.Logs) => {
          console.groupEnd()
          console.timeEnd('Epoch training time')
          console.log(epoch, log)

          const testBatch = data.nextTestBatch(2000)
          const score = model.evaluate(testBatch.xs.reshape([2000, 32, 32, 3]) as any, testBatch.ys as any)
          console.timeEnd('Totol training time')
          score[0].print()
          score[1].print()

          return tf.nextFrame()
        }
      }
    })

  await tf.nextFrame()
}

async function load () {
  const data = new Cifar10()
  await data.load()
  await train(data)

  // const {xs, ys} = data.nextTrainBatch(1500)
  // console.log(xs, ys)
}

load()
