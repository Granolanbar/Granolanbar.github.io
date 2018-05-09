---
title: "Class Day 1 Python Syntax"
layout: post
date: 2017-08-14
image: /assets/images/markdown.jpg
headerImage: false
tag:
- markdown
- elements
star: true
category: blog
author: Nolan Werner
description: Day 1 Notes
---

Heading

These are my notes using Markdown


```python
# Add decimal point to keep fraction when dividing

5/2.0
```




    2.5




```python
print "Day 1 of class print function"
```

    Day 1 of class print function



```python
# Determine type of variable

a = 2
type(a)
```




    int




```python
b = 2.45
type(b)
```




    float




```python
c = True
d = False
print type(c)
print type(d)
```

    <type 'bool'>
    <type 'bool'>



```python
type("String")
```




    str




```python
str1 = "String 1"
str2 = "String 2"
str_combined = str1 + " " + str2
print str_combined
```

    String 1 String 2



```python
# Percent sign gives you the remainder when dividing integers

10 % 4
```




    2




```python
# Use Double Asterics for Exponents

10 ** 2
```




    100




```python
2**4
```




    16




```python
# Scientific Notation

1e4
```




    10000.0




```python
9e4
```




    90000.0




```python
12.45e6
```




    12450000.0




```python
12.45E6
```




    12450000.0



Code Code code


```python
my_Name = "Nolan"
print "My name is " + my_Name
```

    My name is Nolan



```python
print "I am",my_Name
```

    I am Nolan



```python
print "I am {}".format(my_Name)
```

    I am Nolan



```python
Albert = "Albert"
print "Hello {0}, I am {1}".format(Albert,my_Name)
```

    Hello Albert, I am Nolan



```python
print "Hello {a}, I am {b}".format(a=Albert,b=my_Name)
```

    Hello Albert, I am Nolan



```python
print "Hello {0}, I am {1}".format("Albert","Nolan")
```

    Hello Albert, I am Nolan



```python
print "Hello {banana}".format(banana=my_Name)
```


```python
prepared = ""
while prepared != "Y" and prepared != "N":
    prepared = raw_input("Hello, are you ready to order? (Y/N)")
```


```python
if prepared == "Y":
    order = ""
    while order != "Fish Burger" and order != "Big Mac" and order != "Salad":
        order = raw_input("Would you like a Fish Burger, Big Mac, or Salad?")
    if order == "Fish Burger" or order == "Big Mac":
        upsize = raw_input("Would you like to upsize the fries and drink with your {}? (Y/N)".format(order))
        if upsize == "Y":
            print "Okay. That will be a {} with upsized fries and drink. Your food will be ready soon. Please come again!".format(order)
        else:
            print "Okay. That will be a {} with small fries and drink. Your food will be rady soon. Please come again!".format(order)
    elif order == "Salad":
        print "Sounds good! Your total price will be $2.75."
    else:
        print "Please ensure you spelled your order correctly."
else:
    print "Okay. Please let me know when you are ready to order."
```

    Would you like a Fish Burger, Big Mac, or Salad?sald
    Please ensure you spelled your order correctly.



```python

```
