from sympy import symbols,exp
import sympy
import graphviz
import random
import copy

class util():
    def __repr__(self):
        return str(self.value)
    
    def simplify(self, expr):
        expr = sympy.expand(expr)
        expr = sympy.factor(expr)
        expr = sympy.simplify(expr)
        return expr

class spec():

    def sigmoid(self,x):
        return 1/(1+exp(-x))
    def total_cost(self,E=[],O=[]):
        out = 0
        for i in range(len(O)):
            out = out + (E[i] - O[i])**2
        return out
    
    af = sigmoid
    cf = total_cost
    lr = 5

    def __init__(self,af_string=None,af=None,cf_string=None,seed=14):
        if(af_string=='sigmoid'):
            self.af = self.sigmoid
        if(af != None):
            self.af = af
        if(cf_string=='total_cost'):
            self.cf = self.total_cost

        # Set the seed
        random.seed(42)
        

    
    
class dendrites(spec,util):

    weight = 0
    axon = None
    weight_name = ''
    value = 0

    def __init__(self,prev_neuron,weight_name):
        self.axon = prev_neuron
        self.weight = symbols(weight_name)
        self.weight_name = weight_name
        self.value = self.weight

    def ws(self):
        return self.axon.value*self.weight

class neuron(spec,util):

    value = 0
    link=[]
    bias = 0
    layer_index = 'l'
    name = ''
    raw_value = 0
    
    def __init__(self,prev_link,bias_name,layer_num=None,Name=''):
        
        self.layer_index = f'l{layer_num}'
        self.link = prev_link
        self.bias = symbols(bias_name)
        self.name = Name
        dot.node(self.name, label=self.name+'\\n'+bias_name)

        #Build forward propagation expression
        for i in self.link:
            self.value = self.value + (i.ws())
            dot.edge(i.axon.name, self.name,label=i.weight_name)

        #Use the activation function
        self.raw_value = self.value+self.bias
        self.value = self.af(self.raw_value)

    def back(self,cf):
        temp = symbols('temp')
        self.grad = ((cf.evalf(subs={self.value:temp})).diff(temp)).evalf(subs={temp:self.value})
        temp = symbols('temp')
        self.grad = self.grad*(((self.value.evalf(subs={self.raw_value:temp})).diff(temp)).evalf(subs={temp:self.raw_value}))
        temp = symbols('temp')
        self.grad = self.grad*(self.raw_value.diff(self.bias))

class input_neuron(spec,util):

    value = 0
    layer_marker = 'i'
    name = ''
    
    def __init__(self,value_name):
        
        self.value = symbols(value_name)
        self.name = value_name
        dot.node(self.name, label=self.name)

class output_neuron(spec,util):

    value = 0
    raw_value = 0
    link=[]
    bias = 0
    layer_index = 'o'
    name = ''
    grad = 0
    
    def __init__(self,prev_link,bias_name,Name=''):
        
        self.link = prev_link
        self.bias = symbols(bias_name)
        self.name = Name
        dot.node(self.name, label=self.name+'\\n'+bias_name)

        #Build forward propagation expression
        for i in self.link:
            self.value = self.value + (i.ws())
            dot.edge(i.axon.name, self.name,label=i.weight_name)

        #Use the activation function
        self.raw_value = self.value+self.bias
        self.value = self.af(self.raw_value)

    def back(self,cf):
        temp = symbols('temp')
        self.grad = ((cf.evalf(subs={self.value:temp})).diff(temp)).evalf(subs={temp:self.value})
        temp = symbols('temp')
        self.grad = self.grad*(((self.value.evalf(subs={self.raw_value:temp})).diff(temp)).evalf(subs={temp:self.raw_value}))
        temp = symbols('temp')
        self.grad = self.grad*(self.raw_value.diff(self.bias))

        
class network(spec,util):

    inputs = {'count':0,'list':{}}
    weights = {'count':0,'list':{}}
    bias = {'count':0,'list':{}}
    outputs = {'count':0,'list':{}}
    expected = {'count':0,'list':{}}
    network = []
    backprop = {}

    cost_function = None

    def backprop_pass(self,Inputs=[],Expected=[]):
        
        c=0
        for i in self.inputs['list']:
            self.inputs['list'][i]=Inputs[c]
            c=c+1
        c=0
        for i in self.expected['list']:
            self.expected['list'][i]=Expected[c]
            c=c+1
        
        
        #You have to change this, this calculates the cost, you have to calculate the grad here using the dict cost_function {w:eq}
        
        w = list(self.weights['list'].keys())
        b = list(self.bias['list'].keys())

        for i in w:
            temp = copy.deepcopy(self.weights['list'])
            temp.pop(i)
            expr = self.cost_function.evalf(subs={**self.inputs['list'],**self.expected['list'],**temp,**self.bias['list']}).diff(i)
            self.backprop[i] = expr.evalf(subs=self.weights['list'])

            self.weights['list'][i] = self.weights['list'][i] - (self.lr*self.backprop[i])
        
        for i in b:
            temp = copy.deepcopy(self.bias['list'])
            temp.pop(i)
            expr =self.cost_function.evalf(subs={**self.inputs['list'],**self.expected['list'],**self.weights['list'],**temp}).diff(i)
            self.backprop[i] = expr.evalf(subs=self.bias['list'])

            self.bias['list'][i] = self.bias['list'][i] - (self.lr*self.backprop[i])

            
    def build_input(self,n):

        layer_array = []

        for i in range(n):
            temp = input_neuron(f'I{i+1}')
            self.inputs['list'][temp.value]=None
            layer_array.append(temp)

        self.inputs['count'] = n
        return layer_array

    def build_layer(self,layer_number,previous_layer,n):

        layer_array = []

        for i in range(n):
                
                c = 0
                temp_array=[]

                for j in previous_layer:
                    self.weights['count'] = self.weights['count'] + 1
                    con = dendrites(j,f'W{layer_number}{i+1}{c+1}')
                    self.weights['list'][con.weight]=random.random()
                    temp_array.append(con)
                    c=c+1

                node = neuron(temp_array,f'B{layer_number}{i}',layer_number,Name=f'L{layer_number}{i}')
                self.bias['count'] = self.bias['count'] + 1
                self.bias['list'][node.bias]=random.random()
                layer_array.append(node)

        return layer_array
    
    def build_output(self,previous_layer,n):
        layer_array = []

        for i in range(n):
                
                c = 0
                temp_array=[]

                for j in previous_layer:
                    self.weights['count'] = self.weights['count'] + 1
                    con = dendrites(j,f'Wo{i+1}{c+1}')
                    self.weights['list'][con.weight]=random.random()
                    temp_array.append(con)
                    c=c+1

                node = output_neuron(temp_array,f'Bo{i}',Name=f'O{i}')
                self.bias['count'] = self.bias['count'] + 1
                self.outputs['count'] = self.outputs['count'] + 1
                self.outputs['list'][node.value]=None
                self.bias['list'][node.bias]=random.random()
                layer_array.append(node)

        return layer_array
    
    def show(self,straight_line=False):
        if(straight_line==True):
            dot.attr('graph', splines='line')
        dot.render(format='png',view=True)

    def __repr__(self):
        self.show()
        return 'Graph Drawn Succesfully'

    def __init__(self,struct):

        global dot
        dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})

        self.cost_function = self.cf

        self.input_count = struct[0]
        self.output_count = struct[-1]
        for i in range(len(struct)):
            if(i==0):
                self.network.append(self.build_input(self.input_count))
            elif (i==len(struct)-1):
                self.network.append(self.build_output(self.network[-1],self.output_count))
            else:
                if(struct[i]==0):
                    continue
                else:
                    self.network.append(self.build_layer(i,self.network[-1],struct[i]))

        temp_E = []
        temp_O = []
        for i in range(len(self.network[-1])):
            temp_E.append(symbols(f'E{i}'))
            temp_O.append(self.network[-1][i].value)
            self.expected['count'] = self.expected['count'] + 1
            self.expected['list'][temp_E[-1]]=None
        
        self.cost_function = self.cf(temp_E,temp_O)

        


sr = network([2,2,1])
print(sr.weights)
for i in range(100):
    sr.backprop_pass([1,1],[1])
print("After : ")
print(sr.weights)



