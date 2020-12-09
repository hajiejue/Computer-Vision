classdef test < matlab.unittest.TestCase
%     Test your challenge solution here using matlab unit tests
%     
%     Check if your main file 'challenge.m', 'disparity_map.m' and 
%     verify_dmap.m do not use any toolboxes.
%     
%     Check if all your required variables are set after executing 
%     the file 'challenge.m'   



% For the test you have to choose the path twice, first select the path to scene folder
% Second time select the ground truth file.
    properties(TestParameter)
        
        functions = {'challenge.m','disparity_map.m','verify_dmap.m'};
    end
    
    properties
        
         vars = cell(1);
        
    end
    
    methods (TestClassSetup)    
        
        function ClassSetup(test)
            challenge;
            test.vars{1} = D;
            test.vars{2} = R;
            test.vars{3} = T;
            test.vars{4} = G;
        end
        
    end
%     
    methods (Test,ParameterCombination = 'sequential')     
        %check toolboxes:Überprüfen die Dateien challenge.m, disparity_map.m und 
        %verify_dmap.m, ob wirklich keine Toolbox verwendet wird
        function check_toolboxes(test,functions)
           [~,pList] = matlab.codetools.requiredFilesAndProducts(functions);
           verifyEqual(test,pList.Name,'MATLAB');

        end
        %check variables: Überprüfen Sie, dass alle geforderten Variablen in der Datei
        %challenge.m nicht leer bzw. größer als Null sind.
        function check_variables(test)          
            [~,n] = size(test.vars); 
            for i = 1:n
                verifyNotEmpty(test,test.vars{i});
            end
        end
        %check psnr: Vergleichen Sie ihre eigene Implementierung des PSNR mit der 
        %in der Image Processing Toolbox und prüfen Sie, ob das Ergebnis innerhalb 
        %einer angemessenen Toleranz liegt.
        function check_psnr(test)  
            selbstpsnr = verify_dmap(test.vars{1}, test.vars{4});
            toolboxpsnr = psnr(uint8(test.vars{1}), uint8(test.vars{4}));
            verifyLessThan(test,abs(selbstpsnr-toolboxpsnr),0.1);            
        end
        
    end
    
end