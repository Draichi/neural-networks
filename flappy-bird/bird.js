// regex to convert es5 functions to es6 classes
// find: this\.(.*?)\s*=\s*function
// replace: $1

function mutate(x) {
  if (random(1) < 0.1) {
    let offset = randomGaussian() * 0.5;
    let newx = x + offset;
    return newx;
  } else {
    return x;
  }
}

class Bird {
  constructor(brain) {
    this.y = height/2
    this.x = 64
    this.gravity = .6
    this.velocity = 0
    this.lift = -7
    this.score = 0
    this.fitness = 0
    if (brain) {
      this.brain = brain.copy()
      this.brain.mutate(mutate);
    } else {
      this.brain = new NeuralNetwork(
        4, // inputs
        4, // hidden layers
        2  // output
      )
    }
  }
  copy() {
    return new Bird(this.brain);
  }

  show () {
    stroke(255)
    fill(255, 100)
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
    if (output[0] > output[1]) {
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