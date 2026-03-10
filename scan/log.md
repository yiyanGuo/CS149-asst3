test exclusive scan performance

1. version1: simple cuda

| Array Size | Student GPU Time (ms) |
|------------|-----------------------|
| 1,000,000  | 0.729                 |
| 10,000,000 | 7.069                 |
| 20,000,000 | 13.617                |
| 40,000,000 | 26.970                |

2. version2: block scan and add pre-block sum

| Array Size | Student GPU Time (ms) |
|------------|-----------------------|
| 1,000,000  | 0.804                 | 
| 10,000,000 | 3.237                 |
| 20,000,000 | 4.609                 |
| 40,000,000 | 8.161                 |