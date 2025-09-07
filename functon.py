#Q:1
def sum_of_evens(numbers):
    total=0
    for n in numbers:
        if n % 2==0:  #check if even
            total+=n
            return total
        #example usage
        nums=[1,2,3,4,5,6,7,8]
        print("sum of even numbers:",sum_of_evens(nums))


#Q:2
def reverse_string(text):
    return text[::-1]
#example usage
word="hello"
print("reversed string:",reverse_string(word))

#Q:3
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



#Q:7
def power_of_two(n):
    for i in range(n+1):
        yield 2**i #generater power of 2


       
      #example:powers of 2 up to exponent 4
for value in power_of_two(4):
    print(value)


#Q:8
#list of tuples
data=[(1,4),(3,1),(5,9),(2,6)]

#sort using lambda (by second element)
sorted_data=sorted(data,key=lambda x:x[1])
print("original list:",data)
print("sorted list:",sorted_data)
                                        

#Q:9
#celsius to fahrenheit conversion:F=(c*9/5)+32

celsius=[0,20,37,100]        # list of celsius temperature
fahrenheit=list(map(lambda c:(c*9/5)+32,celsius))                            

print("Celsius:",celsius)
print("Fahrenheit:",fahrenheit)


#Q:10
#remove vowels using filter()
text="Anjali is Confident Girl"

#VOWEL LIST
vowels="aeiouAEIOU"

#filter out vowels
result="".join(filter(lambda ch:ch not in vowels,text))
print("original string",text)
print("without Vowels:",result)

#Q:11
#INPUT LIST:[ordernumber,tittle,Quantity,price per item]
orders=[[34587,"learning python, mark lutz",4,40.95],[98762,"programming python,mark lutz",5,56.80],
        [77226,"head firdt python,paul barry",3,32.95],[88112,"einfuhrung in python3,Bernd klein",3,24.99]]

#use map + lambda to calculate result
result=list(map(lambda item:(item[0],round(item[2]*item[3]+(10 if (item[2]*item[3])<100 else 0),2)),orders))

print(result)

                                        