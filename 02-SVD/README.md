### Singular Value Decomposition

## 1. Cơ sở toán học liên quan:

* Vector x và giá trị \lambda lần lượt được gọi là vector riêng và trị riêng của ma trận A khi: Ax = \lambda x

* Một hệ cơ sở u = {u1, u2, ..., un} được gọi là một hệ trực giao (orthogonal) khi các vector trong hệ cơ sở u đó là các vector khác vector 0 và tích vô hướng của 2 vector khác nhau bất kì bằng 0. 

* Một hệ cơ sở u = {u1, u2, ..., un} được gọi là một hệ trực chuẩn (orthonormal) khi nó là một hệ trực giao và độ dài Euclidean (norm 2) của các vector trong hệ bằng 1. 

## 2. Singular Value Decomposition 

* Phương pháp Dimensionality Reduction sử dụng Phân rã ma trận (Matrix Factorization/Decomposition)

* Ý tưởng: biểu diễn 1 ma trận A thành các ma trận nhỏ hown: A = U*S*V.T 

Trong đó:

* U, V là ma trận được tạo bởi 2 hệ trực chuẩn U, V.

* S: ma trận chéo (trong 1 số trường hợp, ma trận S không vuông nhưng vẫn tạm gọi là ma trận chéo).

## 3. Tính chất của SVD 

* A = U*S*V.T và A.T = V*S*U.T

* A.T*A = V*S*U.T*U*S*V.T

* A.T*A = V*S*S*V.T

* A.T*A*V = V*S*S

