### Singular Value Decomposition

## 1. Cơ sở toán học liên quan:

* Vector x và giá trị \lambda lần lượt được gọi là vector riêng (eigenvector) và trị riêng (eigenvalue) của ma trận A khi: Ax = \lambda x

* Một hệ cơ sở u = {u1, u2, ..., un} được gọi là một hệ trực giao (orthogonal) khi các vector trong hệ cơ sở u đó là các vector khác vector 0 và tích vô hướng của 2 vector khác nhau bất kì bằng 0. 

* Một hệ cơ sở u = {u1, u2, ..., un} được gọi là một hệ trực chuẩn (orthonormal) khi nó là một hệ trực giao và độ dài Euclidean (norm 2) của các vector trong hệ bằng 1. 

## 2. Singular Value Decomposition 

* Phương pháp Dimensionality Reduction sử dụng Phân rã ma trận (Matrix Factorization/Decomposition)

* Ý tưởng: biểu diễn 1 ma trận A thành các ma trận nhỏ hown: A = U*S*V.T 

Trong đó:

* U, V là ma trận được tạo bởi 2 hệ trực chuẩn U, V.

* S: ma trận chéo (trong 1 số trường hợp, ma trận S không vuông nhưng vẫn tạm gọi là ma trận chéo).

## 3. Tính chất của SVD 

* A = U*S*V.T và A.T = V*S.T*U.T

* A.T*A = V*S.T*U.T*U*S*V.T = V*S.T*S*V.T

* A.T*A = V*S*S*V.T

* A.T*A*V = V*S*S

* A * V = U * S

## 4. Cách tìm U, S, V:

Cho ma trận A bất kì

* B1: Tìm A.T * A

* B2: Tìm trị riêng và vector riêng của kết quả trong B1. 

* B3: Cho det(A.T * A - \lambda * I) = 0. Tìm ra \lambda 

* B4: Căn bậc hai của \lambda là các phần tử trên đường chéo của ma trận S. => Tìm ra S.

* B5: Ứng với \lambda, tìm trị riêng và vector riêng của các ma trận A.T * A - \lambda * I. Từ đó tìm ra các singular vector của V. => Tìm ra V.

* B6: Dùng tính chất (5) ở phần 3, tìm ra U. 

## 5. Ứng dụng 

* Giảm chiều dữ liệu (Image Compression là ví dụ).

* Cơ sở của các phương pháp giảm chiều dữ liệu khác (Truncated SVD, PCA, v.v.).

## Tài liệu tham khảo:

https://www.youtube.com/watch?v=mBcLRGuAFUk

https://www.youtube.com/watch?v=cOUTpqlX-Xs