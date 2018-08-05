# How to contribute
  - Tensorflex is a library still in it's nascent stages, so we really appreciate third-party pull requests for improving it. 
  - There are only very minimal guidelines that need to be followed by developers before submitting code.

## Guidelines
  - Please open an issue if extant functionality is not working as expected with __proper usage (or failure) examples__, and __mention the current Tensorflow C API version you are running along with your OS__. 
  - If you would like to submit a pull request as a fix for opened issues, refer to the issue in your PR and ensure that adding your code does not break any other tests.
  - If you are submitting a PR with new functionality, please write documentation adhering to the existing documentation style (read the docs [here](https://hexdocs.pm/tensorflex/Tensorflex.html)), and also write tests describing this functionality. You can also open an issue for new functionality to discuss your ideas with the contributors/maintainers.  
  - If your PR contains C code please format it using clang-format: `clang-format -i <your-C-code-file.c>`
  
## Key contribution areas
Tensorflex basically involves four main functionalities that can be improved upon:
  - Improving code for matrices, linear algebra, and numeric computation through NIFs __[Intermediate]__
  - Writing/using Python examples for pre-existing graphs (or creating your own graph models) and running them in Tensorflex, replete with the entire prediction pipeline __[Beginner]__
  - Porting over the Tensorflow C API's functionality in the form of NIFs that _adds_ value to Tensorflex __[Intermediate]__
  - Adding low-level C code that will help improve upon the functionality missing in the Tensorflow C API and then porting those to Tensorflex using NIFs __[Advanced]__

## Getting started
  - Pick any one of the four contribution areas, depending on your comfort level with NIFs, Elixir, or Tensorflow, and think of a contribution. Feel free to discuss your ideas with any of the contributors beforehand by opening an issue for new functionality.
  - Fork the repository, and then clone it.
  - Add the functionality locally, make sure tests pass (`mix test`), and write (then view) documentation (`mix docs`). If all is fine, push those commits to your branch and submit a PR to `master`.
  
