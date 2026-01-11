Open the terminal in the project folder(alternatively one can """cd("<project folder>")""").
Install Julia 1.11.7(I recommend to install julia through [JuliaUp](https://github.com/JuliaLang/juliaup))

> juliaup add 1.11.7
> julia +1.11.7

Enter the Package Manager mode by pressing ']', then run:

> activate .

Then run 'instantiate' to install all the requirements:

> instantiate

Backslash to exit Package Manager mode and then run:

> include("script.jl")
