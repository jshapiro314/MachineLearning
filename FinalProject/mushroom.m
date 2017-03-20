
fileID = fopen('mushroom.txt');
formatSpec = '%s';
C = textscan(fileID,formatSpec);
row = size(C{1},1);
col = 23;


% G is the data matrix
G = zeros(row,col-1);

% L is the label vector, 1 for male, -1 for female
L = zeros(row, 1);

for i = 1:row
    a = strread(C{1}{i}, '%s', 'delimiter', ',');
    for j = 1:col
       A(i,j) = str2double(a{j,1});
    end    
end

% G is the data.remove 11th column (most data missing), and 17th
% column (all entries are the same), first column is the label
G = horzcat(A(:,1:16),A(:,18:col));
G = horzcat(G(:,1:10),G(:,12:col-1));
G = G(:,2:col-2);
 
% labels, label for edible -1, for poisonous 1
L = A(:,1);
for i = 1:row
    if L(i,1) == 0
        L(i,1) = -1;
    end
end

% separate into training and test data 80/20
c = ceil(0.8*row);
trainGt = G(1:c,:);
testG = G(c+1:row,:);
trainLt = L(1:c,:);
testL = L(c+1:row,:);




