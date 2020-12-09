function [D, R, T] = disparity_map(scene_path)
% This function receives the path to a scene folder and calculates the
% disparity map of the included stereo image pair. Also, the Euclidean
% motion is returned as Rotation R and Translation T.
addpath(scene_path);
im0=imread('im0.png');
im1=imread('im1.png');
[M,N]=size(im0);
% Use the size of image to determine which image shall be calculated
switch M
    case 1988
      % Determine the window size of census algorithm and the searching
      % range for corresponding points d.
       window_sizex = 25;
       window_sizey = 25;
       d = 250;
       cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1];
       % sigma and k_p are the parameters for filter.
       sigma=0.6;
       k_p=5;
    case 490
       window_sizex = 7;
       window_sizey = 7;
       d = 20;
       cam0=[542.019 0 541.836; 0 542.019 255.198; 0 0 1];
       sigma=0.6;
       k_p=5;
    case 1956
       window_sizex = 25;
       window_sizey = 25;
       d = 370;
       cam0=[7125.31 0 2290.766; 0 7125.31 946.763; 0 0 1];
        sigma=0.6;
       k_p=5;
    case 434
       window_sizex = 11;
       window_sizey = 11;
       d = 16;
       cam0=[711.499 0 376.135; 0 711.499 227.447; 0 0 1];
        sigma=0.8;
       k_p=3;
end
D=disparity_d(im0,im1,window_sizex,window_sizey,d,sigma,k_p); 
[R T]=RT_mainfunction(im0,im1,cam0);


function [R T]=RT_mainfunction(left_image,right_image,K_cam_0)
% This function calculates the the euclidean transformation

% Rgb to Gray
gray_left_image  = rgb_to_gray(left_image);
gray_right_image = rgb_to_gray(right_image);


% Brechnung der Gross der Bild
[heigth width] =size(gray_right_image);

% Markmalpunkte
segment_length = 15;
k = 0.05;
tau = 1000000;
do_plot = 1;
min_dist = 20;
N = 50;
tile_size = [200 100];

merkmale_1 = harris_detektor(gray_left_image,segment_length, k ,tau,do_plot,min_dist,tile_size,N);
merkmale_2 = harris_detektor(gray_right_image,segment_length, k ,tau,do_plot,min_dist,tile_size,N);

% Korrespondenzen
window_length = 25;
min_corr = 0.95;
do_plot = 0;

Korrespondenzen = punkt_korrespondenzen(gray_left_image,gray_right_image,merkmale_1,merkmale_2,window_length,min_corr,do_plot);
% Kalibierungsmatrix
K_cam_0 = [3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1];
%K_cam_1 = [3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1];
%K_cam_0 = [3.997684 0 1.176728; 0 3.997684 1.011728; 0 0 1];
% RanSaC Eingabeparameter 
epsilon = 0.5;
p = 0.5;
tolerance = 0.01;
%Korrespondenzen_robust = F_ransac(Korrespondenzen, epsilon,p,tolerance);
% Essentielle_matrix
Essentielle_matrix = achtpunktalgorithmus(Korrespondenzen, K_cam_0);  
disparity = Korrespondenzen(1,1:end) - Korrespondenzen(3,1:end)   ;   


[T1, R1, T2, R2, U, V]=TR_aus_E(Essentielle_matrix);
[T R lambda] = rekonstruktion(T1, T2, R1, R2, Korrespondenzen, K_cam_0);
[row col] = size(Korrespondenzen);
x1 = [Korrespondenzen(1:2,:) ; ones(1,col)];
x2 = [Korrespondenzen(3:4,:) ; ones(1,col)];
x1 = inv(K_cam_0)*x1;
x2 = inv(K_cam_0)*x2;
P1=(ones(3,1)*lambda(:,1)').*x1;




function merkmale = harris_detektor(input_image,segment_length, k ,tau,do_plot,min_dist,tile_size,N)
% This function calculate the gradient of the image to determing the
% corresponding points of the two images
[heigth width h] = size(input_image);
    if h==1
         X = double(input_image);
         [Ix, Iy] = sobel_xy(X);
         %w = fspecial ('gaussian',[segment_length,1],segment_length/3);
         w=[0.0345378632877979;0.0447931940270943;0.0558157567658479;0.0668235931358567;0.0768654283296716;0.0849494359953811;0.0902024157483115;0.0920246254200779;0.0902024157483115;0.0849494359953811;0.0768654283296716;0.0668235931358567;0.0558157567658479;0.0447931940270943;0.0345378632877979];
         G11=conv2(w,w,Ix.^2,'same');
         G22=conv2(w,w,Iy.^2,'same');
         G12=conv2(w,w,Ix.*Iy,'same'); 
         G =[G11 G12,G12 G22];
    end
    
     for i = 1:heigth 
         for j = 1:width 
          R = [G11(i,j),G12(i,j);G12(i,j),G22(i,j)];       
          H(i,j) = det(R)-k*(trace(R))^2;          
     
         end
     end
     
    for i=1:heigth
        for j=1:width
            if H(i,j)>=tau
                corners(i,j)=H(i,j);
            else
                corners(i,j)=0;
            end
        end
    end
    Cake = cake(min_dist);
    [JJ,LL]= size(corners);
  
    Neu = zeros(JJ+2*min_dist,LL+2*min_dist); 
    Neu(min_dist+1:JJ+min_dist,min_dist+1:LL+min_dist) = corners;
    corners = Neu;
    [B,I] = sort(corners(:),'descend');
    I(B==0)=[];
    sorted_index = I;
    
    [GG,VV] = size(sorted_index);
    II = ceil(heigth/tile_size(1));
    UU = ceil(width/tile_size(2));
 
    AKKA = zeros (II,UU);
    
    if (II*UU*N>=GG)
        merkmale = zeros(2,GG);
    else
        merkmale = zeros(2,II*UU*N);
    end   
    
    [XX,YY] = size(sorted_index);
    [ZZ,PP] = size(corners);
    JJ=1;
    
    for  i = 1:XX

    j = sorted_index(i);
  
    if(corners(j)==0)
            continue;
    else
           col = floor(j/ZZ);
           row = j - col*ZZ;
           col = col + 1;
    end

    Ex = floor((row-min_dist-1)/(tile_size(1)))+1;
    Ey = floor((col-min_dist-1)/(tile_size(2)))+1;
    
    AKKA(Ex,Ey)=AKKA(Ex,Ey)+1;
    corners(row-min_dist:row+min_dist,col-min_dist:col+min_dist)=corners(row-min_dist:row+min_dist,col-min_dist:col+min_dist).*cake(min_dist);
    if AKKA(Ex,Ey)==N
        corners((((Ex-1)*tile_size(1))+1+min_dist):min(size(corners,1),Ex*tile_size(1)+min_dist),(((Ey-1)*tile_size(2))+1+min_dist):min(size(corners,2),Ey*tile_size(2)+min_dist))=0;   
    end
    
    merkmale(:,JJ)=[col-min_dist;row-min_dist];
    JJ = JJ+1;
    end
    merkmale = merkmale(:,1:JJ-1);
    
 function Cake = cake(min_dist)
 
        [X,Y]=meshgrid(-min_dist:min_dist,[-min_dist:-1,0:min_dist]);
        Cake=sqrt(X.^2+Y.^2)>min_dist;

end
end

function Korrespondenzen = punkt_korrespondenzen(gray_left_image,gray_right_image,merkmale_1,merkmale_2,window_length,min_corr,do_plot)
    [I1_XX,I1_YY] = size(gray_left_image);
    [I2_XX,I2_YY] = size(gray_right_image);
    [Mpt1_XX,Mpt1_YY] = size(merkmale_1);
    [Mpt2_XX,Mpt2_YY] = size(merkmale_2);
    Im1=double(gray_left_image);
    Im2=double(gray_right_image);
    for i = 1:Mpt1_YY
        if  merkmale_1(1,i) < ceil(window_length/2) 
            merkmale_1(1,i) = 0;
        end
        if  merkmale_1(1,i) > I1_YY - floor(window_length/2)
            merkmale_1(1,i) = 0;
        end
    end

    for j = 1:Mpt1_YY
        if merkmale_1(2,j) < ceil(window_length/2)  
            merkmale_1(2,j) = 0;
        end   
        if merkmale_1(2,j) > I1_XX - floor(window_length/2)
            
            merkmale_1(2,j) = 0;
        end
    end
    
    for p = 1:Mpt2_YY
        if  merkmale_2(1,p) < ceil(window_length/2) 
            merkmale_2(1,p) = 0;
        end
        if merkmale_2(1,p) > I2_YY - floor(window_length/2)
            merkmale_2(1,p) = 0;
        end
    end
    for q = 1:Mpt2_YY
        if merkmale_2(2,q) < ceil(window_length/2)  
            merkmale_2(2,q) = 0;
        end
        if merkmale_2(2,q) > I2_XX - floor(window_length/2)
            merkmale_2(2,q) = 0;
        end
    end  
    
    id=merkmale_1(1,:)==0 ;
    merkmale_1(:,id)=[];
    id=merkmale_1(2,:)==0 ;
    merkmale_1(:,id)=[];

    id=merkmale_2(1,:)==0 ;
    merkmale_2(:,id)=[];
    id=merkmale_2(2,:)==0 ;
    merkmale_2(:,id)=[];
    
    [~,no_pts1] = size(merkmale_1);
    [~,no_pts2] = size(merkmale_2); 
    
    Mat_feat_1 = zeros(window_length*window_length,no_pts1);
    Mat_feat_2 = zeros(window_length*window_length,no_pts2);
    

   for i= 1:no_pts1
        
        % Zentrum und Grenze fuer Mpt1
        LinkMpt1  = merkmale_1(1,i)-floor(window_length/2);    
        RechtMpt1 = merkmale_1(1,i)+floor(window_length/2);
        
        ObenMpt1  = merkmale_1(2,i)-floor(window_length/2);    
        UntenMpt1 = merkmale_1(2,i)+floor(window_length/2);
        % Suchen den Ort im Originalbild
        NichtNormierungsPunkt1 = Im1( ObenMpt1: UntenMpt1,LinkMpt1:RechtMpt1);
        
        % Normierung
        Mat_feat_1(:,i) = (NichtNormierungsPunkt1(:)-mean(NichtNormierungsPunkt1(:))) / std(NichtNormierungsPunkt1(:));
      
    
   end
        % Die Verfahrung ist gleich wie Mpt1
   for j= 1:no_pts2
        
        LinkMpt2  = merkmale_2(1,j)-floor(window_length/2);    
        RechtMpt2 = merkmale_2(1,j)+floor(window_length/2);
        
        ObenMpt2  = merkmale_2(2,j)-floor(window_length/2);    
        UntenMpt2 = merkmale_2(2,j)+floor(window_length/2);
        
        NichtNormierungsPunkt2 = Im2( ObenMpt2: UntenMpt2,LinkMpt2:RechtMpt2);
    
       Mat_feat_2(:,j) = (NichtNormierungsPunkt2(:)-mean(NichtNormierungsPunkt2(:))) / std(NichtNormierungsPunkt2(:));
   end

    N = window_length * window_length ;
    % NCC Brechnung
    Temporaere_Matrix = 1/(N-1)*Mat_feat_2.'* Mat_feat_1;
    % Unterhalb des Schwellwerts min_corr 
    Temporaere_Matrix(Temporaere_Matrix < min_corr) = 0;
    
    NCC_matrix = Temporaere_Matrix;
     
    % die Indizes aller Korrespondenzwerte in NCC_matrix die nicht 0 sind
    [BB,II] = sort(NCC_matrix(:),'descend');
    II(BB==0)=[];
    % Speichern
    sorted_index = II;
    
    
    
    Korrespondenzen=zeros(4,min(no_pts1,no_pts2));
    % Spaltenvektoranzeige
    index_in_Korrespondenzen = 1 ;
 
    length_sorted_index = size(sorted_index);
       
    
    for i= 1:length_sorted_index 
        %  ob es bereits vorhanden ist
        if(NCC_matrix(sorted_index(i))==0)
            
                 continue;
        else 
        % Suchen den ursprünglichen Speicherort 
        [index_in_Mpt2,index_in_Mpt1]=ind2sub(size(NCC_matrix),sorted_index(i));
        end
        %ein Merkmalspunkt in Bild 1 nicht mehr als einem anderen Merkmalspunkt in Bild 2 zugewiesen werden kann
        NCC_matrix(:,index_in_Mpt1) = 0;
        % speichern korrespondierende Punkte
        Korrespondenzen(:,index_in_Korrespondenzen) = [merkmale_1(:,index_in_Mpt1);merkmale_2(:,index_in_Mpt2)];  
        index_in_Korrespondenzen = index_in_Korrespondenzen+1;
       
       
    end
        
    Korrespondenzen = Korrespondenzen(:,1:index_in_Korrespondenzen - 1);
   

end

function [Essentielle_matrix] = achtpunktalgorithmus(Korrespondenzen_robust, K_cam_0)
   x1=Korrespondenzen_robust(1:2,:);
   x2=Korrespondenzen_robust(3:4,:);
   length_x1=size(x1,2);
   length_x2=size(x2,2);
   switch nargin 
       % Berechnung der Fundamentalmatrix
        case 1
          x1=[x1(1:2,:);ones(1,length_x1)];
          x2=[x2(1:2,:);ones(1,length_x2)];
        case 2
       % Berechnung der essentiellen Matrix   
          x1=[x1(1:2,:);ones(1,length_x1)];
          x2=[x2(1:2,:);ones(1,length_x2)];
          x1=inv(K_cam_0)*x1;
          x2=inv(K_cam_0)*x2;
        
   end

   for i=1:length_x1
   A(i,:) = kron(x1(:,i),x2(:,i));
   end
   
   [U,S,V]=svd(A);
    Gs = V(:,9);
    G = reshape(Gs,3,3);
    %Singulaerwertzerlegung von G
    [UG,SG,VG]=svd(G);
    Essentielle_matrix = UG*[1 0 0;0 1 0;0 0 0]*VG';

end

function [T1, R1, T2, R2, U, V]=TR_aus_E(Essentielle_matrix)
    [UG,SG,VG]=svd(Essentielle_matrix);

    if UG'*UG == eye(3) & det(UG) == 1 & VG'*VG == eye(3) & det(VG) ==1
        U = UG;
        V = VG;
    else
        U = UG*[1 0 0;0 1 0 ; 0 0 -1];
        V = VG*[1 0 0;0 1 0 ; 0 0 -1];
    end
    R1 = U * [0 -1 0 ;+1 0 0 ;0  0 1 ]' * V';    
    R2 = U * [0  1 0 ;-1 0 0 ;0  0 1 ]' * V';
    T1_dach = U * [0 -1 0 ; +1 0 0  ; 0  0 1 ]* SG * U';            
    T2_dach = U * [0  1 0 ;-1 0 0  ;  0  0 1 ]* SG * U';                 
    T1 =  [-1*T1_dach(2,3) ; T1_dach(1,3) ;T1_dach(2,1)]  ;
    T2 =  [-1*T2_dach(2,3); T2_dach(1,3) ;T2_dach(2,1)]  ;

   
end

function [T R lambda] = rekonstruktion(T1, T2, R1, R2, Korrespondenzen_robust, K)
    T_cell = {[T1] [T2] [T1] [T2]};
    R_cell = {[R1] [R1] [R2] [R2]};
    [row col] = size(Korrespondenzen_robust);

    x1 = [Korrespondenzen_robust(1:2,:) ; ones(1,col)];
    x2 = [Korrespondenzen_robust(3:4,:) ; ones(1,col)];
    x1 = inv(K)*x1;
    x2 = inv(K)*x2;
    
    T = 0;
    R = 0;
    lambda = 0;
    M1 = 0;
    M2 = 0;
    % size der Korrespondenzen
    [row col] = size(Korrespondenzen_robust);

    % Initialisierung der positive lambda
    max_pos_lambda=0;
    % Berechnung der M1 und M2
    for j = 1:4
       
         M1 = 0;
         M2 = 0;
        for i=1:col
            temp_M1_ein(:,i) = cross(x2(:,i),R_cell{1,j}*x1(:,i)); 
            temp_M1_zwei(:,i) = cross(x2(:,i),T_cell{1,j});
            temp_M2_ein(:,i) = cross(x1(:,i),R_cell{1,j}'*x2(:,i)); 
            temp_M2_zwei(:,i) = -1*cross(x1(:,i),R_cell{1,j}'*T_cell{1,j});
                    end
        for i=1:col
                M1((i-1)*3+1:i*3,i) = temp_M1_ein(:,i);
                M2((i-1)*3+1:i*3,i) = temp_M2_ein(:,i);
        end

        M1 = [M1 reshape(temp_M1_zwei,[],1)] ;
        M2 = [M2 reshape(temp_M2_zwei,[],1)] ;
        % Berechnung der d1 und d2
        [UG_d1,SG_d1,VG_d1]=svd(M1);
        [UG_d2,SG_d2,VG_d2]=svd(M2);
        d1 =  VG_d1(:,end) ; 
        d2 =  VG_d2(:,end) ;
        d1=d1/d1(end,1);
        d2=d2/d2(end,1);
        d1(end,:) = [];
        d2(end,:) = [];  
        d_cell{j} = {[d1 d2]};
        % Bechnung der max_pos_lambda R T und besteste lambda
        if sum(sum([d1 d2]>0)) >  max_pos_lambda
            
             max_pos_lambda = sum(sum([d1 d2]>0));
             R = R_cell{1,j};
             T = T_cell{1,j};
             lambda = [d1 d2];
        end
       
        
end
            
    
   

end


end

   
    
end

function [gray_image] = rgb_to_gray(rgb_image)
    X=double(rgb_image);
    [m,n,h] = size(X);
if h==3
    gray =zeros(m,n);
        for i = 1:m
           for j = 1:n
           gray(i,j) = 0.299*X(i,j,1) +0.587*X(i,j,2)+0.114*X(i,j,3);
           end
        end
end

if h==1
    gray = X;
end
gray_image = uint8(gray);
end

function [Fx, Fy] = sobel_xy(input_image)
soble_x = [1 2 1;0 0 0;-1 -2 -1];
soble_y = soble_x';
Fx=conv2(input_image,soble_y,'same');
Fy=conv2(input_image,soble_x,'same');  
end

function k=disparity_d(image1,image2,window_sizex,window_sizey,d,sigma,k_p)
% We use the Census algorithm to determine the cost value of each point 
% And use this as a basis to find the corresponding points in the two images.
% The explanation of the principle of the algorithm is written in the report.
image_l = rgb_to_gray(image1);
image_r = rgb_to_gray(image2);
[xx,yy] = size(image_l);
% When calculating the Census, we need to take a window for each point, 
% So we need to expand the image according to the size of the window.
imag1 = img_fill(image_l,window_sizex,window_sizey);
imag2 = img_fill(image_r,window_sizex,window_sizey);

disparity = zeros(xx,yy);
 for t = 1:xx
    p = t/xx*100;
    disp([num2str(p),'% has been completed'])
    disparity(t,:) = disp_row_egdfill(imag1,imag2,t,window_sizex,window_sizey,d);
 end
% k=disparity;
 k=med(disparity,k_p);

function disp = disp_row_egdfill( img1, img2 ,x , dx , dy , d )
% Actually calculates the disparity.
% Initialize parameters
    [cen1,cen2] = cen_row(img1,img2,dx,dy, x);
    cen1 = cen1';
    cen2 = cen2';

    cen1(:,end:end+d) = cen1(:,end-d:end);

    disp = zeros(1,length(cen2));

 % Calculate the difference between Hamming distance.
 for kk = 1:size(cen2,2)
            oo = cen1(:,kk:kk+d-1);
            qq = cen2(:,kk);
            pp = xor(oo, qq);
            [~,iiii]=min(sum(pp));
            disp(kk)=iiii;
 end  
end

function [census1 census2] = cen_row(input_image1,input_image2, window_sizex, window_sizey, rr)
    % This function calculates the Census value for each point in the specified line of the image.
    % The input variable window_sizex,window_sizey represents the size of the window when calculating census.
    % The input variable rr represents the row of the picture to be calculated.
    img1 = input_image1;
    img2 = input_image2;
    [~,colnum] = size(img1);
    
    % Determine the location of the current calculation point. 
    offsetx = ceil(window_sizex/2)-1;
    offsety = ceil(window_sizey/2)-1;
    % Calculate the position of the center point based on the window size
    center_x = ceil(window_sizex/2);
    center_y = ceil(window_sizey/2);
    
    census1 = zeros(colnum-2*offsety,window_sizex*window_sizey-1);
    census2 = zeros(colnum-2*offsety,window_sizex*window_sizey-1);
        for j = 1:colnum-2*offsety
            % cx cy are the location of the current calculation point. 
            cx = rr+offsetx;
            cy = j+offsety;
            % Count the element number.
            num1 = 1;
            num2 = 1;
            for a = 1:window_sizex
                for b = 1:window_sizey
                    % Calculate the difference between point in the block
                    % to the center point.
                    value1 = img1(rr-1+a,j-1+b)-img1(cx,cy);
                    value2 = img2(rr-1+a,j-1+b)-img2(cx,cy);
                    if(a~=center_x||b~=center_y)
                        if(value1<=0)
                            census1(j,num1) = 0;
                            num1 = num1+1;
                        else
                            census1(j,num1) = 1;
                            num1 = num1+1;
                        end
                        if (value2<=0)
                            census2(j,num2) = 0;
                            num2 = num2+1;                     
                        else
                            census2(j,num2) = 1;
                            num2 = num2+1;    
                        end
                    end
                end
            end

        end      
end

end
function img_f = img_fill( img_unf, window_sizex, window_sizey)
% When calculating the Census, we need to take a window for each point, 
% So we need to expand the image according to the size of the window.
    img = img_unf;
    [row,col] = size(img);
    
    fillx = floor(window_sizex/2);
    filly = floor(window_sizey/2);
    img_f = ones(row+2*fillx,col+2*filly) .*128;
    img_f = uint8(img_f);
    img_f(fillx+1:end-fillx,filly+1:end-filly) = img;

end
function img = med( image, m )

    n = m;
    [ height, width ] = size(image);
    x1 = double(image);
    x2 = x1;
    for i = 1: height-n+1
        for j = 1:width-n+1
            mb = x1( i:(i+n-1),  j:(j+n-1) );
            mb = mb(:);
            mm = median(mb);
            x2( i+(n-1)/2,  j+(n-1)/2 ) = mm;
 
        end
    end
 
    img = uint8(x2);
end
