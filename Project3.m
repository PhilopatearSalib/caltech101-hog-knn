clc,clear
load('chair_featurevector');    
load('chairTest_featurevector');
load('crab_featurevector');
load('crabTest_featurevector');
load('cellphone_featurevector');
load('cellphoneTest_featurevector');
load('dolphin_featurevector');
load('dolphinTest_featurevector');
load('sunflower_featurevector');
load('sunflowerTest_featurevector');
load('stopsign_featurevector');
load('stopsignTest_featurevector');
load('helicopter_featurevector');
load('helicopterTest_featurevector');
load('lamp_featurevector');
load('lampTest_featurevector');
load('kangaroo_featurevector');
load('kangarooTest_featurevector');
load('car_featurevector');
load('seehorse_featurevector');
load('Motorbikes_featurevector');
load('starfish_featurevector');
load('laptop_featurevector');
load('laptopTest_featurevector');
load('buddha_featurevector');


%k = size(buddah_featurevector,3);
%m = 1;
%i = 21;

TestSampleNN  = zeros(10,size(crabTest_featurevector,3));
TestSampleKNN = zeros(10,size(crabTest_featurevector,3));
% Motor Bike Test Sample
for T = 1:10
    
        if T == 1
        TEST = sunflowerTest_featurevector;
        end
    
        if T == 2
        TEST = crabTest_featurevector;
        end
    
        if T == 3
        TEST = chairTest_featurevector;
        end
    
        if T == 4
        TEST = cellphoneTest_featurevector;
        end
    
         if T == 5
         TEST = dolphinTest_featurevector;
         end
            
        if T == 6
        TEST = sunflowerTest_featurevector;
        end
        
        if T == 7
        TEST = stopsignTest_featurevector;
        end
        
        if T == 8
        TEST = helicopterTest_featurevector;
        end
        if T == 9
        TEST = lampTest_featurevector;
        end
        
        if T == 10
        TEST = kangarooTest_featurevector;
        end
        
for  i = 1 : size(TEST,3)
k = size(sunflower_featurevector,3);
m = 1;
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - sunflower_featurevector(:,:,j-m+1));
        label(j) = 1;
    end
    
    m = k+1;
    k = k + size(crab_featurevector,3);
    
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - crab_featurevector(:,:,j-m+1));
        label(j) = 2;
    end  
    
    m = k+1;
    k = k + size(chair_featurevector,3);
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - chair_featurevector(:,:,j-m+1));
        label(j) = 3;
    end 

    m = k+1;
    k = k + size(cellphone_featurevector,3);
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - cellphone_featurevector(:,:,j-m+1));
        label(j) = 4;
    end 

    m = k+1;
    k = k + size(dolphin_featurevector,3);
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - dolphin_featurevector(:,:,j-m+1));
        label(j) = 5;
    end 
    
    m = k+1;
    k = k + size(sunflower_featurevector,3);
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - sunflower_featurevector(:,:,j-m+1));
        label(j) = 6;
    end 
    
    m = k+1;
    k = k + size(stopsign_featurevector,3);
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - stopsign_featurevector(:,:,j-m+1));
        label(j) = 7;
    end

    m = k+1;
    k = k + size(helicopter_featurevector,3);
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - helicopter_featurevector(:,:,j-m+1));
        label(j) = 8;
    end 
    
    m = k+1;
    k = k + size(lamp_featurevector,3);
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - lamp_featurevector(:,:,j-m+1));
        label(j) = 9;
    end
    
    m = k+1;
    k = k + size(kangaroo_featurevector,3);
    for j = m : k
        dist(j) = norm(TEST(:,:,i) - kangaroo_featurevector(:,:,j-m+1));
        label(j) = 10;
    end

[sorted_dist order] = sort(dist);
TestSampleNN(T,i) = label(order(1));
NNlabel (i,1) = TestSampleNN(T,i);
k = size(chair_featurevector,2);
m = 1;
TestSampleKNN(T,i)= mode(label(order(1:5)));
end
end