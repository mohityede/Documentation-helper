# Documentation-helper
Creating Q&amp;A like chatbot for documentation.

#### Notes
- I have created synchronous function `add_batch` to add batch
- `GoogleGenerativeAIEmbeddings` don't have parameter `retry_min_time` hence asynchronous batch insertion is not possible through it.