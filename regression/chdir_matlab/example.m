
% The example.txt is a gene expression file. In the file there are a control 
% group of 20 replicates and an experiment group of 6 replicates. This script 
% shows how to calculate the characteristic direction vector from the data
% using the chdir module.
% 
% Author: Qiaonan Duan
% Ma'ayan Lab, Icahn School of Medicine at Mount Sinai
% Oct. 15, 2013
% 
clear all;close all;


z = importdata('example.txt');
genes = z.rowheaders(2:end);

data = z.data(2:end,:);
mark = z.data(1,:);

ctrlIdx = mark==0;
expmIdx = mark==1;

% unitV is the characteristic direction and its absolute component values 
% are sorted in descending order. Each component value in unitV matches a 
% gene in genes.
[CD,genes] = chdir(data(:,ctrlIdx),data(:,expmIdx),genes,1);
foldchange=mean(data(:,expmIdx),2)./mean(data(:,ctrlIdx),2);
pvalue=[];
for i=1:size(genes,1)
    [h,p]=ttest2(data(i,ctrlIdx),data(i,expmIdx));
    pvalue=[pvalue;p];    
end
