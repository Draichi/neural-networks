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
    this.brain = new NeuralNetwork()
  }

  show () {
    fill(255)
    ellipse(this.x, this.y, 25, 25)
  }

  up () {
    this.velocity =+ this.lift

  }

  update () {
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