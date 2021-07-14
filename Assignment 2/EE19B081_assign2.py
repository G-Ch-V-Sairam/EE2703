'''The command to be given on terminal is 
python3 EE19B081_assign2.py the_req_netlist_file'''


import math
import numpy as np
from sys import argv,exit

if len(argv)!=2:					#Checking whether the user gave only 1 input file.
	print("Give only one input netlist file.")
	exit()

CKT='.circuit'
END='.end'			#Giving variable names to them is better than hardcoding them everywhere.
AC='.ac'
	
class Resistor:
	def __init__(self,name,node1,node2,value):		#A new class , Resistor
		self.name=name
		self.node1=node1
		self.node2=node2					#Define its name , from node , to node and its value
		self.value=value    
	

class Capacitor:
	def __init__(self,name,node1,node2,value):		#A new class , Capacitor
		self.name=name
		self.node1=node1
		self.node2=node2
		if ac>0:
			self.value=complex(0,-1/(w*value))	#If ac source , then impedance=1/j*w*C
		else:
			self.value=1e100					#If dc source , it is equivalent to open circuit
		
		
class Inductor:
	def __init__(self,name,node1,node2,value):		#A new class , Inductor
		self.name=name
		self.node1=node1
		self.node2=node2
		if ac>0:
			self.value=complex(0,w*value)		#If ac source , impedance=j*w*L
		else:
			self.value=1e-100					#If dc source , it is equivalent to short circuit
        
			
class Source:
	def __init__(self,name,node1,node2,value,element):		#A new class , Source , both voltage and current sources
		self.name=name
		self.node1=node1
		self.node2=node2
		self.value=value
		self.element=element

def append_nodes(node_num):
	if node_num not in nodes:					#We must not add a node into this list again if it already exists.
		nodes.append(node_num)
	return nodes.index(node_num)

def integer_nodes(n):				#Changes any alphanumeric node names to numericals.
	if n=="GND":
		return 0
	else:
		n=int(str(n)[-1])
		return n


nodes =[] #list of Node names
resistors = []
capacitors = []
inductors = []						#Keep track of the different elements in the circuits by creating lists
voltage_sources = []
current_sources = []
nodes.append(integer_nodes("GND"))					#Adding the node "GND" to the list of node names


try:
	with open(argv[1]) as f:
		lst=f.readlines()				#Open the file and read its lines
	start=-1;end=-2;ac=-1
	for each_line in lst:
		if each_line[:len(CKT)] == CKT:			#If a line begins with .circuit , then start=index of that line
			start=lst.index(each_line)
		elif each_line[:len(END)] ==END:
			end=lst.index(each_line)			#Similarly for .end and .ac
		elif each_line[:len(AC)] == AC:
			ac=lst.index(each_line)
		
	
	if start>=end or (ac>=0 and ac<end):		#The '.circuit' must come before '.end' and '.ac' must come after '.end'
		print("Invalid circuit file")			#Any netlist file not following this is an invalid circuit
		exit()
	
	if ac>0:
		w=float(lst[ac].split()[2])
		print("Frequency:",(w))					#If it is an ac circuit , then note its frequency
		if w<=0:
			print("This frequency is not possible.It should be dc")
			exit()
		else:
			w=2*math.pi*(w)
		
	for line in lst[start+1:end]:
		write=line.split('#')[0].split()		#Split the lines , delete the comments
		l=len(write)
		
		if l==4:								#If l=4 , then the element must be either R,L or C
			element=write[0]					#Name of the element
			node1=integer_nodes(write[1])						#from node
			node2=integer_nodes(write[2])						#to node
			value=float(write[3])				#Value of the element
			
			from_node=append_nodes(node1)
			to_node=append_nodes(node2)			#Add these nodes into the list of node objects if they are'nt already present.
			
			if element[0]=='R':
				x=Resistor(element,node1,node2,value)
				resistors.append(x)							#Add the element into the list of resistors
			elif element[0]=='L':
				x=Inductor(element,node1,node2,value)
				inductors.append(x)							#Add this element into the list of inductors
			elif element[0]=='C':
				x=Capacitor(element,node1,node2,value)
				capacitors.append(x)						#Add this element into the list of capacitors
				
			else:
				print("Invalid element")				#If the element is neither R,L or C , then the netlist file is wrong.
				exit()
						
		elif l==6 and write[3]=='ac':			#If l=6 , then it must be an ac voltage or current source
			element=write[0]
			node1=integer_nodes(write[1])
			node2=integer_nodes(write[2])
			value=float(write[4])/(2)				#Since we need the rms value
			phase=float(write[5])
			
			from_node=append_nodes(node1)			#Append the nodes into the list
			to_node=append_nodes(node2)
			
			x=Source(element,node1,node2,complex(value*math.cos(phase),value*math.sin(phase)),element[0])
			
			if element[0]=='V':
				voltage_sources.append(x)			#Append this element into the list of voltage sources
			elif element[0]=='I':
				current_sources.append(x)			#Append this element into the list of current sources
			else:
				print("This element dos not exist.")
				
		elif l==6 and write[3]!='ac':
			print("This file does not work for circuits containing dependent sources.")		
			exit()
			
		elif l==5 and ac>0:
			print("Malformed input file.")			
			exit()
			
		elif l==5 :								#If l=5 , then it is a dc source.
			element=write[0]
			node1=integer_nodes(write[1])
			node2=integer_nodes(write[2])							
			value=float(write[4])
			
			from_node=append_nodes(node1)			#Append the nodes into the list.
			to_node=append_nodes(node2)				
			
			x=Source(element,node1,node2,value,element[0])
			if element[0]=='V':
				voltage_sources.append(x)			#Add the element into the list of voltage sources
			elif element[0]=='I':
				current_sources.append(x)			#Add the element into the list of current sources.
			else:
				print("This element does not exist")
				exit()
		else:
			print("Malformed input file")
			exit()
			
	#Constructing the matrices M,b
	
	size=len(nodes)+len(voltage_sources)					#The matrix M has the same number of rows and columns and it is equal to numbr of nodes + voltage sources 
	if ac>0:
		M=np.zeros((size,size),dtype=complex)			#Complex values also should be appendable in the case of ac circuit.
		b=np.zeros(size,dtype=complex)	
	else:
		M=np.zeros((size,size))						#No need of complex values if there are no ac sources.
		b=np.zeros(size)
	
						#NODAL ANALYSIS Matrix forming
	M[0][0]=1
	for r in resistors:
		M[r.node1][r.node1] +=1/r.value
		M[r.node1][r.node2] -=1/r.value
		M[r.node2][r.node1] -=1/r.value
		M[r.node2][r.node2] +=1/r.value
	
	
	for l in inductors:
		M[l.node1][l.node1] += 1/l.value
		M[l.node1][l.node2] -= 1/l.value
		M[l.node2][l.node1] -= 1/l.value
		M[l.node2][l.node2] += 1/l.value
	
	for c in capacitors:
		M[c.node1][c.node1] += 1/c.value
		M[c.node1][c.node2] -= 1/c.value
		M[c.node2][c.node1] -= 1/c.value
		M[c.node2][c.node2] += 1/c.value
		
	for v in voltage_sources:
		M[v.node1][voltage_sources.index(v)+len(nodes)] -=1
		M[v.node2][voltage_sources.index(v)+len(nodes)] +=1
		M[voltage_sources.index(v)+len(nodes)][v.node1] -=1
		M[voltage_sources.index(v)+len(nodes)][v.node2] +=1
		b[voltage_sources.index(v)+len(nodes)]=v.value
	
	for i in current_sources:		
		b[i.node1] +=i.value
		b[i.node2] -=i.value
	
	X=np.linalg.solve(M,b)					#Solves the matrix equation Mx=b
	
	i=1
	for n in nodes[1:]:
		print("Voltage of node",n," :Magnitude = ",np.abs(X[i]) ,"Phase = ",math.degrees(np.angle(X[i])))		#Gives magnitude and phase of Voltages of nodes
		i+=1
	
	for v in voltage_sources:
		print("Current through voltage source",v.name,":Magnitude = ",np.abs(X[i])," Phase = ",math.degrees(np.angle(X[i])))		#Gives magnitude and phase of current through voltage sources.
		i+=1
		
except IOError:
	print("Malformed input file.")	
	exit()
