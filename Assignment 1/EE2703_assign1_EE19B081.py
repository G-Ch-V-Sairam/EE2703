'''THE COMMAND LINE TO BE GIVEN IN THE UBUNTU TERMINAL TO RUN THIS PROGRAM IS 
		
		python3 assgn1_final.py the_given_netlist_file				'''

from sys import argv,exit										
if len(argv)!=2:								#Checking whether the user actually gave only the required number of inputs
	print("\n You need to give only one file as input")
	exit()
	
start='.circuit\n'								#Giving variable names to these strings is better than hard-coding them
end1='.end'														
end2='.end\n'
	
try:
	with open(argv[1]) as f:					#Taking the file given on command line as input and it is given to the variable f
		lst=f.readlines()						#Gives a list of all the lines in f
	
	for _ in range(0,len(lst)):					#Deleting all the lines till the line containing '.circuit'
		if lst[0]!=start:
			del lst[0]
		else:
			del lst[0]
			break
	for __ in range(0,len(lst)):				#Deleting all the lines after and including '.end'
		if lst[-1]!=end1 and lst[-1]!=end2:
			del lst[-1]
		else:
			del lst[-1]
			break
	
	'''1.The file may not contain either .circuit or .end or both.
	   2.The .end line may be above the .circuit line
	   In these cases , the lines 17-32 deletes all the items in the list and gives us an empty list.So, as below, we get rid of these 2 errors.'''
	    	
	if lst==[]:											 
		print("Invalid circuit definition\n")			
		exit(0)											
	
	#Deleting the comments in the file,if any.-	
	for num in range(len(lst)):							
		for nums in range(len(lst[num])):
			if lst[num][nums]=='#':					
				lst[num]=lst[num][:nums]+'\n'
				break
	
	#The file may contain multiple '.circuit's or '.end's.This error is gotten rid of in the below lines. 
	for ___ in lst:
		if ___==start or ___==end1 or ___==end2:				
			print("Invalid circuit definition.Your circuit has more than one '.circuit' or '.end' files.")
			exit(0)
		
		#If any of the lines in the file contain only a comment , the above code gives us a '/n'.We delete all these empty lines as below.
		if ___=='\n':							
			lst.remove('\n')						
	#So the list lst would contain the main body of the file , i.e; the part of the file between .circuit and .end , divided into lines, after deleting all comments.
	
	#Create an empty list and append the strings in the main body in reverse order.
	new_list=[]
	for m in range(0,len(lst)):							
		new_list.append(lst[m].split())
		new_list[m].reverse()
	new_list.reverse()
	#The list new_list contains the tokens given in the file in the reverse order in which we would print out in the next step.

	in_reverse=''
	for line in new_list:
		for word in line:
			in_reverse=in_reverse+word+' '
		in_reverse=in_reverse+'\n'							#Printing the tokens in reverse order
	
	print(in_reverse)

except IOError:
	print("Invalid file")
	exit()	
