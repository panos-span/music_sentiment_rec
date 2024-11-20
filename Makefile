# Makefile for compiling LaTeX documents

# Name of the main LaTeX file (without extension)
MAIN = main

# LaTeX and BibTeX commands
LATEX = pdflatex
BIBTEX = bibtex

# Directory for bibliography style files
BST_DIR = bst

# Default target
all: pdf

# Target to generate the PDF
pdf: $(MAIN).tex
	TEXINPUTS=.:$(BST_DIR): $$TEXINPUTS $(LATEX) $(MAIN).tex
	TEXINPUTS=.:$(BST_DIR): $$TEXINPUTS $(BIBTEX) $(MAIN)
	TEXINPUTS=.:$(BST_DIR): $$TEXINPUTS $(LATEX) $(MAIN).tex
	TEXINPUTS=.:$(BST_DIR): $$TEXINPUTS $(LATEX) $(MAIN).tex

# Clean up auxiliary files
clean:
	sudo rm -f *.aux *.bbl *.blg *.log *.out *.toc

# Clean up all generated files, including the PDF
cleanall: clean
	sudo rm -f $(MAIN).pdf

.PHONY: all pdf clean cleanall
