const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

var raw_dataset = require('./iris')
var dataset =[]
var label =[]
raw_dataset.map((value)=>{
    dataset.push([value.sepalLength , value.sepalWidth , value.petalLength , value.petalWidth])
    if(value.species == 'setosa'){label.push([1,0,0])}
    else if(value.species == 'versicolor'){label.push([0,1,0])}
    else {label.push([0,0,1])}
})

console.log(dataset)
console.log(label)

var x = tf.tensor2d(dataset)
var y = tf.tensor2d(label)

var model = tf.sequential()

model.add(tf.layers.dense({
    inputShape: 4,units:4,
    activation: 'relu'
}))

model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax'
}))

model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'sgd',
    metrics: ['accuracy']
})

model.fit(x,y,{
    epochs: 50,
    shuffle: true,
    batchSize: 16
})

model.evaluate(x,y).print()