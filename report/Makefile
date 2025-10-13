COMPILER = pdflatex
BIB = bibtex

SRCS = Report.tex

all: build

build: $(SRCS)
	$(COMPILER) -shell-escape $<
	$(COMPILER) -shell-escape $<

clean:
	rm -f *.aux
	rm -f *.log
	rm -f *.out
	rm -f *.pdf
	rm -f *.toc

.PHONY: all build clean