# **STAR - Utilizing Physics-Informed Neural Networks to Approximate Nonlinear PDEs**
Researcher: **Tobias Safie** | [tks57@drexel.edu](mailto:tks57@drexel.edu) | [LinkedIn](linkedin.com/in/tsafie) | [GitHub](github.com/tobiassafie)
<br>
Advisors  : Dr. Niharika Sravan, Natalya Pletskova
## **Purpose**
The purpose of this project is to explore the accuracy, efficiency, and usages of deep learning to approximate solutions for computationally expensive systems and equations. 
Physics-Informed Neural Networks (PINNs) being used to solve nonlinear partial differential equations (PDEs) in physics is not a brand new idea, but, of course, it can be improved upon, which is a goal of this project.
<br><br>
We first did this by feeding our neural networks toy models of kilinova evolution. 
<br><br>
The meat and potatoes of this project, though, comes from the application in quantitative finance— especially options pricing.
The science behind the valuation of options lies in complex, stochastic nonlinear PDEs that are frankly incredibly computationally expensive and time consuming to approximate via computational integration methods or solve analytically, of course. 
The most known and 'base' equation is the Black-Scholes equation, a PDE that was created to evaluate the pricing of traditional European options.
<p align="center">
  <strong>Black-Scholes PDE</strong><br>
  <code>∂V/∂t + (1/2)σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0</code>
</p>
Since then, that equation has evolved into more complex forms that includes more variables and derivatives, as to try to account for the volatility and external factors of the market.
I attempted to approximate the Heston PDE, a modern, American options evaluation PDE which heavily builds off the Black-Scholes equation.
<p align="center">
  <strong>Heston PDE</strong><br>
  <code>
    ∂V/∂t + (1/2) v S² ∂²V/∂S² + ρσv S ∂²V/∂S∂v + (1/2) σ²v ∂²V/∂v² + rS ∂V/∂S + κ(θ − v) ∂V/∂v − rV = 0
  </code>
</p>
The Heston PDE is inaproximable without being simplified into toy models by typical numerical methods— but these toy models are still incredibly computationally expensive.
The solution to this issue is PINNs, which can account for both the stochastic calculus and the 4-dimensionality.
