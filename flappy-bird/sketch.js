var bird
var pipes = []
function setup () {
	createCanvas(400, 500)
	bird = new Bird()
	pipes.push(new Pipe())
}

function draw () {
	background(0)

	for (var i=0; i<pipes.length; i++) {
		pipes[i].show()
		pipes[i].update()

		if (pipes[i].hits(bird)) {
			console.log("HIT")
		}

		if (pipes[i].offscreen()) {
			pipes.splice(i, 1)
		}
	}

	bird.update()
	bird.show()

	if (frameCount % 123 == 0) {
		pipes.push(new Pipe())
	}
	
	
}

function keyPressed () {
	if (key == ' ') {
		bird.up()
	}
}