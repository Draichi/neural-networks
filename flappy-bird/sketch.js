const TOTAL = 1000
var birds = []
var pipes = []
function setup () {
	createCanvas(640, 480)
	for (let i=0; i<TOTAL; i++) {
		birds[i] = new Bird()
	}
	pipes.push(new Pipe())
}

function draw () {
	background(0)

	for (let i=pipes.length-1; i>=0; i--) {
		pipes[i].show()
		pipes[i].update()

		for (let j=birds.length-1; j>=0; j--) {
			if (pipes[i].hits(birds[j])) {
				birds.splice(j, 1)
			}
		}

		if (pipes[i].offscreen()) {
			pipes.splice(i, 1)
		}
	}

	for (let bird of birds) {
		bird.think(pipes)
		bird.update()
		bird.show()
	}


	if (birds.length === 0) {
		nextGeneration()
	}

	if (frameCount % 123 == 0) {
		pipes.push(new Pipe())
	}
}

// function keyPressed () {
// 	if (key == ' ') {
// 		bird.up()
// 	}
// }