# Documentation-helper
Creating Q&amp;A like chatbot for documentation.

#### Notes
- we are making `add_batch` function async
- after calling async funtion in for loop(list comperhansion) we gether all output into `results` using `asyncio.gather()`