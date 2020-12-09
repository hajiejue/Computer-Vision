function psnr = verify_dmap(D, G)
% This function calculates the PSNR of a given disparity map and the ground
% truth. The value range of both is normalized to [0,255].
 MSE =(double(G) -double(D)).^2;
    b = mean(mean(MSE));
    
    psnr = 10*log10(255^2/b);
end

