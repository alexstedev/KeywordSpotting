all: build run

build:
	docker build -t kws .

run:
	docker run -it --rm -p 8501:8501 kws
