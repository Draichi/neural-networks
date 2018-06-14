// regex to convert es5 functions to es6 classes
// find: this\.(.*?)\s*=\s*function
// replace: $1

class Bird {
  constructor() {
    this.y = height/2
    this.x = 64
    this.gravity = .6
    this.velocity = 0
    this.lift = -7
    this.score = 0
    this.fitness = 0
    this.brain = new NeuralNetwork(
      4, // inputs
      4, // hidden layers
      1  // output
    )
  }

  show () {
    fill(255)
    ellipse(this.x, this.y, 25, 25)
  }

  up () {
    this.velocity =+ this.lift

  }

  think(pipes) {

    // find the closest pipe
    let closestPipe = null
    let closestPipeDistance = Infinity
    for (let i=0; i<pipes.length; i++) {
      let distance = pipes[i].x - this.x
      if (distance < closestPipeDistance && distance > 0) {
        closestPipe = pipes[i]
        closestPipeDistance = distance
      }
    }

    let inputs = []
    inputs[0] = this.y / height
    inputs[1] = closestPipe.top / height
    inputs[2] = closestPipe.bottom / height
    inputs[3] = closestPipe.x / width

    let output = this.brain.predict(inputs)
    if (output[0] > 0.5) {
      this.up()
    }
  }

  update () {
    this.score ++
    this.velocity += this.gravity
    this.y += this.velocity

    if (this.y > height) {
      this.y = height
      this.velocity = 0
    }
    if (this.y < 0) {
      this.y = 0
      this.velocity = 0
    }
  }
}