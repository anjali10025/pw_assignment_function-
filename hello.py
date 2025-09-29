def reverse_string(text):
    return text[::-1]
#example usage
word="hello"
print("reversed string:",reverse_string(word))



def countSquares(x):
    sqrt = x**0.5
    result = int(sqrt)  
    return result

# driver code
x = 9
print(countSquares(x))


#Q:4
def is_prime(num):
    if num<2:
        return False
    for i in range(2,int(num*0.5)+1):#check divisibility
        if num%i==0:
            return False
        return True
            
        #check no. from 1 to 200
    for n in range(1,201):
        if is_prime(n):
            print(n,"is prime")
        else:
            print(n,"is not prime")



a=15;b=3;c=4
calc=a+b*c//(c%b)-5 
print(calc)
    

a=5;b=3
print(a&b)



d={1:"python",2:[1,2,3]}
print(d.update({"one:2"}))