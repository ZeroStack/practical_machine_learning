# Set working directory to the folder file is contained in
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Load relevant libraries
library(knitr)
library(markdown)

knit("project.Rmd")
markdownToHTML("project.md", "project.html", options = c("use_xhml"))
