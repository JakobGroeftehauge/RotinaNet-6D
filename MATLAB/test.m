clear;
pred = [1,2,3,4,5,6,7,8,9];

matlab_ortho = reshape(pred,[3,3])';
[U,S,V] = svd(matlab_ortho);
d = det(V*U');
matlab_ortho = V*[1,0,0;0,1,0;0,0,sign(d)]*U';
ortho_pred = matlab_ortho;