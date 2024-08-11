#include <stdio.h>
#include <stdlib.h>

struct Person {
  int age;
  char* name;
};


int main() {
  struct Person person1;
  person1.age = 25;
  person1.name = "elliot";

  printf("age: %d\tname: %s\n", person1.age, person1.name);

  struct Person* person2 = malloc(sizeof(struct Person));
  person2->age = 20;
  person2->name = "not elliot";

  printf("age: %d\tname: %s\n", person2->age, person2->name);
  return 0;
}

