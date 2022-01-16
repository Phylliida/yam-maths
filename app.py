"""
python "C:\Program Files (x86)\Google\google_appengine\appcfg.py" -A yam-maths update .
python "C:\Program Files (x86)\Google\google_appengine\dev_appserver.py" .
"""

from importlib import reload
import warnings
import codecs
import sys
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


from flask import request, render_template, flash, url_for, redirect, Flask
from flask_cors import CORS, cross_origin

from flask import request, redirect, stream_with_context, Response

app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"

import json

import sympy

@app.route("/yam-maths/reference")
@cross_origin()
def reference():
  referencePage = """
<html>
<head>
</head>
<body>
<h2>Notation</h2>
Any variable you type is valid, for example x, y, yams and wow can all be used as variables in the input box.<br /><br />
You can use sqrt(-1) to get i.<br /><br />
Infinity is denoted oo (two lowercase o's)<br /><br />
pi will be interpreted as pi=3.1415...<br /><br />
x**2 means x squared, x^2 is invalid syntax.<br /><br />
You need to have times between all terms, otherwise an error will be thrown.<br />
For example, 2x(y) is not valid, while 2*x*y is. Also, 2*xy will consider "xy" as a single variable, so make sure to seperate variables as needed.<br /><br /> 
(expression).diff(var) is the notation for writing an expression differentiated with respect to variable var. For example:<br />
(x^2+y*x).diff(x)<br />
denotes the derivative of x^2+y*x with respect to x.

<h2>Evaluating equations</h2>
If you want to plug in values into an expression, simply do <br />
(x + y + z).subs(x, 3) <br /><br />
This will replace all instances of x with a 3. If you do something like<br />
(x+y+z).subs(x, 3).subs(y, 4).subs(z, 10)<br />
then the Evaluated box below output will show a number, because all you have left is an expression with lots of numbers in it.<br />

<h2>Solving equations</h2>
Type something of the form:<br />
x = x*3 + 2*x**2<br />
then press the solve button, and the solutions will appear in an arraw below Output, as well as in the Formatted box.<br /><br />
For example, if you type x = x*3 + 2*x**2 then press solve, the values of x that satisfy your equation will be in an array below output, and in the Formatted box, where x = [-1, 0] means that -1 or 0 are solutions for x.<br /><br />
You can also solve differential equationsm via entering in something like <br />
dsolve(f(x).diff(x)*x + y)< br />
Then pressing simplify.


<h2>Press simplify to evaluate</h2>

Find Greatest Common Denominator<br />
gcd(4, 6)<br /><br />
Find Least Common Multiple <br />
lcm(4, 6)<br /><br />
Factor polynomial <br />
factor(x**2 + 2*x + 1) <br /><br />
Expand polynomial<br >
expand((x+y)**4)<br /><br />
Binomial, aka n choose k<br />
binomial(10, 4)<br /><br />
nth catalan number<br />
catalan(100)<br /><br />

<h3>Stats</h3>
<h4>Discrete random variables</h4>
Uniform Distribution<br />
DiscreteUniform(name, items)<br />
DiscreteUniform('yams', [10, 20, 33])<br /><br />
Die<br />
Die(name, sides)<br />
Die('yams', 7)<br /><br />
Bernoulli Distribution<br />
Bernoulli(name, probOfSuccess, succ=1, fail=0)<br />
Bernoulli("yams", 0.5)<br />
Bernoulli("yams", 0.4, succ=1, fail=-1)<br /><br />
Coin toss<br />
Coin(name, p=1/2)<br />
Coin("yams")<br />
Coin("yams", p=0.3)<br /><br />
Binomial distribution<br />
Binomial(name, numTrials, probOfSuccess, succ=1, fail=0)<br />
Binomial("yams", 10, 0.4)<br />
Binomial("yams", 20, 0.2, succ=10, fail=1)<br /><br />
Hypergeometric distribution<br />
Hypoergeometric(name, N, m, n)<br /><br />
Finite Random Variable<br />
FiniteRV(name, densityMap)<br />
FiniteRV("yams", {0: .1, 1: .2, 2: .3, 3: .4})<br /><br />'
<h4>Continuous random variables</h4>
Arcsin distribution<br />
Arcsin(name, a=0, b=1)<br />
a is the left interval boundary, and b is the right interval boundary<br /><br />
Benini Distribution<br />
Benini(name, alpha, beta, sigma)<br /><br />
Beta distribution<br />
Beta(name, alpha, beta)<br /><br />
Beta prime distribution<br />
BetaPrime(name, alpha, beta)<br /><br />
Cauchy distribution<br />
Cauchy(name, x0, gamma)<br /><br />
Chi distribution<br />
Chi(name, k)<br /><br />
Non-central Chi distribution<br />
ChiNoncentral(name, degreesOfFreedom, shiftParamater)<br /><br />
Chi-squared distribution<br />
ChiSquared(name, degreesOfFreedom)<br /><br />
Dagum distribution<br />
Dagum(name, p, a, b)<br /><br />
Erlang distribution<br />
Erlang(name, k, lambda)<br /><br />
Exponential distribution<br />
Exponential(name, lambda)<br /><br />
F distribution<br />
FDistribution(name, d1, d2)<br /><br />
Fisher's Z distribution<br />
FisherZ(name, d1, d2)<br /><br />
Frechet distribution<br />
Frechet(name, a, s=1, m=0)<br /><br />
Gamma distribution<br />
Gamma(name, k, theta)<br /><br />
Inverse gamma distribution<br />
GammaInverse(name, a, b)<br /><br />
Kumaraswamy distribution<br />
Kumaraswamy(name, a, b)<br /><br />
Laplace distribution<br />
Laplace(name, mu, b)<br /><br />
Logistic distribution<br />
Logistic(name, mu, s)<br /><br />
Log-normal distribution<br />
LogNormal(name, mean, sigmaj)<br /><br />
Maxwell distribution<br />
Maxwell(name, a)<br /><br />
Nakagami distribution<br />
Nakagami(name, mu, omega)<br /><br />
Normal distribution<br />
Normal(name, mean, std)<br /><br />
Pareto distribution<br />
Pareto(name, xm, alpha)<br /><br />
U-quadratic distribution<br />
QuadraticU(name, a, b)<br /><br />
Raised cosine distribution<br />
RaisedCosine(name, mu, s)<br /><br />
Rayleigh distribution<br />
Rayleigh(name, sigma)<br /><br />
Student's t distribution<br />
StudentT(name, nu)<br /><br />
Triangular distribution<br />
Triangular(name, a, b, c)<br /><br />
Uniform distribution<br />
Uniform(name, left, right)<br /><br />
Irwin-Hall distribution<br />
UniformSum(name, n)<br /><br />
von Mises distribution<br />
VonMises(name, mu, k)<br /><br />
Weibull distribution<br />
Weibull(name, alpha, beta)<br /><br />
Wigner semicircle distribution<br />
WignerSemicircle(name, R)<br /><br />
Continuous random variable<br />
ContinuousRV(name, pdf, interval=(-oo, oo))<br /><br />
<h4>Functions</h4>
Probability that a condition is true<br />
P(condition)<br />
P(Die('X', 6)> 4)<br /><br />
Probability that a condition is true given another condition<br />
P(Die('X', 6) < 3, X>2)<br /><br />
Optionally, one can provide P(..., numsamples=None, evalf=True, evaluate=True).<br />
numsamples enables sampling and approximatex the probability with that many samples.<br />
evalf should be set to true if sampling should return a number and not a complex expression.<br />
evaluate is in case of a condinuous systems returning unevaluated integrals








<h3>Hyperbolic functions</h3>
Hyperbolic cosine<br />
cosh(7)<br /><br />
Hyperbolic sine<br />
sinh(9)<br /><br />
Hyperbolic tangent<br />
tanh(4)<br /><br />
Hyperbolic secant<br />
sech(5)<br /><br />
Hyperbolic cosecant<br />
csch(pi*3)<br /><br />
Hyperbolic cotangent<br />
coth(5)<br /><br />
Inverse hyperbolic cosine<br />
acosh(4)<br /><br />
Inverse hyperbolic sine<br />
asinh(4)<br /><br />
Inverse hyperbolic tangent<br />
atahn(4)<br /><br />
Inverse hyperbolic secant<br />
asech(4)<br /><br />
Inverse hyperbolic cosecant<br />
acsch(4)<br /><br />
Inverse hyperbolic cotangent<br />
acoth(4)<br /><br />

<h3>Factorials and gamma functions</h3>
Factorial<br />
factorial(10)<br /><br />
Double factorial<br />
fac2(10)<br /><br />
Gamma function<br />
gamma(10)<br /><br />
Reciprocal of the gamma function<br />
rgamma(10)<br /><br />
Rising factorial or Pochhammer symbol<br />
rf(10, 20)<br /<br />
Falling factorial<br />
ff(6, 2)<br /<br />
Beta function<br />
beta(10, 20)<br /><br />
Generalized incomplete beta function<br />
betainc(3, 4, 0, 6)<br /><br />
Super factorial<br />
superfac(5)<br /><br />
Hyper factorial<br />
hyperfac(10)<br /><br />
Bernes G-function<br />
barnesg(7)<br /><br />
psi(m, z) = Polygamma function of order m of z
psi(3, 4)<br /><br />
digamma(z) is shorthand for psi(0, z)<br />
digamma(10)<br /><br />
nth harmonic number<br />
harmonic(7)<br /><br />

<br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />
yam-maths is an interface to sympy, and much of this is from it's documentation <a href="http://docs.sympy.org/0.7.1/">here</a>.
</body>
</html>
  
  """
  return referencePage

@app.route('/yam-maths/')
@cross_origin()
def hello():
  yamMathsHtmlFile = open("YamMaths.html", "rb")
  yamMathsHtml = yamMathsHtmlFile.read()
  yamMathsHtmlFile.close()
  return yamMathsHtml

from sympy.parsing.sympy_parser import parse_expr, eval_expr
from sympy.core.symbol import Symbol as SympySymbol
from sympy.printing.mathml import mathml

import mpmath
import sympy
import cgi
from sympy.parsing.sympy_parser import standard_transformations,implicit_multiplication_application

transformations = (standard_transformations + (implicit_multiplication_application,))


@app.route("/yam-maths/addConstraint", methods=["POST"])
@cross_origin()
def addConstraint():
  if not "constraintText" in request.form:
    return '{"isError": true, "error": "' + 'No constraint given' + '"}'
  elif not "constraints" in request.form:
    return '{"isError": true, "error": "' + 'No constraints given' + '"}'
  try:
    constraintText = request.form['constraintText']
    constraintExpr = parse_expr(str(constraintText),  evaluate=False, global_dict=moreDict)
    
    constraints = json.loads(request.form['constraints'])
    
    for i, constraint in enumerate(constraints):
      constraints[i] = parse_expr(str(constraint['constraintText']), global_dict=moreDict)
    
    constraints.append(constraintExpr)
    vars = getVariables(constraints)
    
    varStrs = [str(var) for var in vars]
    
    return '{"isError": false, "constraintLatex":' + json.dumps(sympy.printing.latex(constraintExpr)) + ', "variables":' + json.dumps(varStrs) + '}'
  except Exception as e:
    return '{"isError": true, "error": "' + str(e) + '"}'
  

@app.route("/yam-maths/solveConstraints", methods=["POST"])
@cross_origin()
def solveConstraints():
  if not "solver" in request.form:
    return '{"isError": true, "error": "' + 'No solver name given' + '"}'
  elif not "constraints" in request.form:
    return '{"isError": true, "error": "' + 'No constraints given' + '"}'
  try:
    solverName = request.form['solver']
    constraints = json.loads(request.form['constraints'])
    
    for i, constraint in enumerate(constraints):
      constraints[i] = parse_expr(str(constraint['constraintText']), global_dict=moreDict)
    
    vars = getVariables(constraints)
    solution = sympy.lambdify(vars, constraints) 
    
    return '{"isError": false, "constraintLatex":' + json.dumps(sympy.printing.latex(solution)) + '}'
  except Exception as e:
    return '{"isError": true, "error": "' + str(e) + '"}'

  
  

@app.route("/yam-maths/optimization")
@cross_origin()
def optimization():
  optimizationHtmlFile = open("Optimization.html", "rb")
  optimizationHtml = optimizationHtmlFile.read()
  optimizationHtmlFile.close()
  return optimizationHtml


@app.route("/yam-maths/addVariable", methods=["POST"])
@cross_origin()
def addVariable():
  if not "variable" in request.form:
    return "No variable given"
  else:
    variableString = request.form['variable']
    try:
      maths = parse_expr(str(variableString), global_dict=moreDict)
    except Exception as e:
      return '{"isError":true, "error":"' + str(e) + '"}'
    
    return '{"text":' + json.dumps(variableString) + '}'
    
    








def getVariables(maths):
  inputVariables = []
  try:
    atoms = maths.atoms()
  except:
    atoms = []
    
    for item in maths:
      vars = getVariables(item)
      atoms.extend(vars)
  
  for atom in atoms:
    if type(atom) is SympySymbol:
      inputVariables.append(atom)
      
  # Remove duplicates
  inputVariables = list(set(inputVariables))
  
  # Sort by variable name
  inputVariables.sort(key=lambda x: str(x))
  
  return inputVariables


def dintegral(originalMaths, maths, low, high):
  originalLatex = sympy.printing.latex(maths)
    
  inputVariables = getVariables(maths)
  
  funStr = "f(" + ", ".join([str(var) for var in inputVariables]) + ") = " +  str(maths)
  gradFunc = [(var, sympy.integrate(maths, (var, low, high))) for var in inputVariables]
  gradFuncPretty = ["int f d" + str(var) + " = " + str(diff) for (var, diff) in gradFunc]
  gradFunctions = sympy.Matrix([diff for (var, diff) in gradFunc])
  
  return toJson(originalMaths, gradFunctions, prettyStr=funStr + "\n\n" + "\n\n".join(gradFuncPretty))

def dsum(originalMaths, maths, low, high):
  originalLatex = sympy.printing.latex(maths)
    
  inputVariables = getVariables(maths)
  
  funStr = "f(" + ", ".join([str(var) for var in inputVariables]) + ") = " +  str(maths) 
  gradFunc = [(var, mpmath.nsum(lambda val: maths.subs({var: val}), [low, high])) for var in inputVariables]
  gradFuncPretty = ["Sum over " + str(var) + " = " + str(diff) for (var, diff) in gradFunc]
  gradFunctions = sympy.Matrix([diff for (var, diff) in gradFunc])
  
  return toJson(originalMaths, gradFunctions, prettyStr=funStr + "\n\n" + "\n\n".join(gradFuncPretty))
  
def integral(originalMaths, maths):
  originalLatex = sympy.printing.latex(maths)
    
  inputVariables = getVariables(maths)
  
  funStr = "f(" + ", ".join([str(var) for var in inputVariables]) + ") = " +  str(maths)
  gradFunc = [(var, sympy.integrate(maths, var)) for var in inputVariables]
  gradFuncPretty = ["int f d" + str(var) + " = " + str(diff) for (var, diff) in gradFunc]
  gradFunctions = sympy.Matrix([diff for (var, diff) in gradFunc])
  
  return toJson(originalMaths, gradFunctions, prettyStr=funStr + "\n\n" + "\n\n".join(gradFuncPretty))

def solve(originalMaths, maths):
  originalLatex = sympy.printing.latex(maths)
  
  inputVariables = getVariables(maths)
  
  
  try:
    actualMaths = []
    for i, math in enumerate(maths):
      if not i == len(maths)-1:
        actualMaths.append(sympy.Eq(math, maths[i+1]))
    maths = actualMaths
  except:
    pass
      
  
  funStr = "f(" + ", ".join([str(var) for var in inputVariables]) + ") = " +  str(maths)
  gradFunc = [(var, [math for math in sympy.solve(maths, var)]) for var in inputVariables]
  
  gradFuncLatex = [[sympy.printing.latex(diffi) for diffi in diff] for (var, diff) in gradFunc]
  
  
  gradFuncPretty = [str(var) + " = " + str(diff) for (var, diff) in gradFunc]
  gradFuncLatex = "[" + ",".join(["[" + ",".join([str(sympy.printing.latex(diffi)) for diffi in diff]) + "]" for (var, diff) in gradFunc]) + "]"
  gradFuncEval = [[diffi.evalf() for diffi in diff] for (var, diff) in gradFunc]
  
  gradFunctions = [diff for (var, diff) in gradFunc]
  
  
  return toJson(originalMaths, gradFunctions, prettyStr=funStr + "\n\n" + "\n\n".join(gradFuncPretty))

def gradient(originalMaths, maths):
    
  inputVariables = getVariables(maths)
  
  funStr = "f(" + ", ".join([str(var) for var in inputVariables]) + ") = " +  str(maths)
  gradFunc = [(var, sympy.diff(maths, var)) for var in inputVariables]
  gradFuncPretty = ["df/d" + str(var) + " = " + str(diff) for (var, diff) in gradFunc]

  gradFunctions = sympy.Matrix([diff for (var, diff) in gradFunc])
  
  return toJson(originalMaths, gradFunctions, prettyStr=funStr + "\n\n" + "\n\n".join(gradFuncPretty))


def hessian(originalMaths, maths):

  inputVariables = getVariables(maths)
  
  hessiFunc = sympy.hessian(maths, inputVariables)
  
  return toJson(originalMaths, hessiFunc)

import collections

def evaluateF(expr):
  try:
      return expr.evalf()
  except:
    if isinstance(expr, collections.Iterable):
      return [evaluateF(expression) for expression in expr]
    elif isinstance(expr, dict):
      for key in expr.keys():
        expr[key] = evaluateF(expr[key])
      return expr
    else:
      return expr

def simplify(expr):
  try:
    expr.simplify();
    return expr
  except:
    if isinstance(expr, collections.Iterable):
      return [simplify(expression) for expression in expr]
    elif isinstance(expr, dict):
      for key in expr.keys():
        expr[key] = simplify(expr[key])
      return expr
    else:
      return expr

def simplifyInput(originalMaths, maths):
  return toJson(originalMaths, maths)

def toJson(originalMaths, mathsOutput, basicStr=None, prettyStr=None, latexStr=None, evaluatedStr=None):
  
  originalMaths = sympy.printing.latex(originalMaths)
  mathsOutput = simplify(mathsOutput)
  
  if basicStr is None:
    basicStr = str(mathsOutput)
  if prettyStr is None:
    prettyStr = sympy.printing.pretty(mathsOutput)
  if latexStr is None:
    latexStr = sympy.printing.latex(mathsOutput)
  if evaluatedStr is None:
    evaluatedStr = evaluateF(mathsOutput)
  
  return '{"original": ' + json.dumps(str(originalMaths)) + ',"basic": ' + json.dumps(str(basicStr)) + ',"pretty": ' + json.dumps(str(prettyStr)) + ', "latex":' + json.dumps(str(latexStr)) + ',"evaluated": ' + json.dumps(str(evaluatedStr)) + '}'

moreDict = {}
exec('from sympy import *',moreDict)
exec('from sympy.stats import *',moreDict)
exec('from sympy.core.relational import *',moreDict)
  
@app.route("/yam-maths/processMath", methods=["POST"])
@cross_origin()
def processMath():
  try:
    
    if not "type" in request.form:
      return "No process type given"
    elif not "maths" in request.form:
      return "No math given"
    else:
      
      processType = str(request.form['type']).lower()
      #try:
      maths = parse_expr(str(request.form['maths']), global_dict=moreDict)
      mathsOld = parse_expr(str(request.form['maths']), evaluate=False, global_dict=moreDict)
      #except Exception as e:
      #  raise e
      #  return "Error parsing: " + str(e)

      if processType == "simplify":
        return simplifyInput(mathsOld, maths)
      elif processType == "gradient":
        return gradient(mathsOld, maths)
      elif processType == "hessian":
        return hessian(mathsOld, maths)
      elif processType == "integral":
        return integral(mathsOld, maths)
      elif processType == "solve":
        return solve(mathsOld, maths)
      elif processType == "bounded":
        if not "low" in request.form:
          return "No low given"
        elif not "high" in request.form:
          return "No high given"
        else:
          try:
            low = parse_expr(str(request.form['low']),  evaluate=False, global_dict=moreDict)
            high = parse_expr(str(request.form['high']), evaluate=False, global_dict=moreDict)
          except Exception as e:
            return "Error parsing: " + str(e)
          if not "operation" in request.form:
            return "No bounded operation given"
          
          operationType = request.form['operation']
          
          if operationType == "integral":
            return dintegral(mathsOld, maths, low, high)
          elif operationType == "sum":
            return dsum(mathsOld, maths, low, high)
          else:
            return "Unknown bounded operation: " + operationType
          
      else:
        return "Unknown process type: " + processType
  except Exception as e:
    return str(e)
    
    
@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, Nothing at this URL.', 404


@app.errorhandler(500)
def application_error(e):
    """Return a custom 500 error."""
    return 'Sorry, unexpected error: ' + str(e), 500

    
    
    
    
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)