# 01


# 02
## Purpose of `#pragma once`?
    - wiki ⇒ [Pragmaonce](https://en.wikipedia.org/wiki/Pragma_once#:~:text=In%20the%20C%20and%20C,once%20in%20a%20single%20compilation).
    - include `#pragma once` so that file is only included once. otherwise you get `error: redefinition of 'foo'` in the code example below

`grandparent.h`
```cpp
// #pragma once

struct foo 
{
    int member;
};
```

`parent.h`

```cpp
#include "grandparent.h"
```

`child.h`

```cpp
#include "grandparent.h"
#include "parent.h"

int main() {

  int member;

}
```