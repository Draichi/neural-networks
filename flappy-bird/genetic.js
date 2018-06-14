function nextGeneration() {
  calculateFitness()
  for (let i=0; i<TOTAL; i++) {
    birds[i] = new Bird()
  }
}

function calculateFitness() {
  let sum = 0
  for (let bird of birds) {
    sum += bird.score
  }
  for (let bird of birds) {
    bird.fitness = bird.score / sum
  }
}