a. `static_cast<new_type>(expression)`


Example:
```cpp
double pi = 3.14159;
int rounded_pi = static_cast<int>(pi);
```

Explanation:
static_cast is used for implicit conversions between types. It's the most common cast and is checked at compile-time. In this example, it converts a double to an int, truncating the decimal part.


b. `dynamic_cast<new_type>(expression)`


Example:
```cpp
class Base { virtual void foo() {} };
class Derived : public Base { };

Base* base_ptr = new Derived;
Derived* derived_ptr = dynamic_cast<Derived*>(base_ptr);
```

Explanation:
dynamic_cast is used for safe downcasting in inheritance hierarchies. It performs a runtime check and returns nullptr if the cast is not possible. It requires at least one virtual function in the base class.


c. `const_cast<new_type>(expression)`


Example:
```cpp
const int constant = 10;
int* non_const_ptr = const_cast<int*>(&constant);
*non_const_ptr = 20; // Modifies the const variable (undefined behavior)
```

Explanation:
const_cast is used to add or remove const (or volatile) qualifiers from a variable. It's the only C++ style cast that can do this. However, modifying a const object leads to undefined behavior.


d. `reinterpret_cast<new_type>(expression)`


Example:
```cpp
int num = 42;
char* char_ptr = reinterpret_cast<char*>(&num);
```

Explanation:
reinterpret_cast is the most dangerous cast. It can convert between unrelated types, like pointers to integers or vice versa. It's often used for low-level operations and should be used with extreme caution.

