%% Initialization

% Closing of all graphs, clearing of all variables and command window
close all;
clc;
commandwindow;

tol = 0.0000001;
iter = 0;
A = [3./5, 4./5;
  -4./5, 3./5];
B = [3./5, 4./6;
  -24./25, 3./5];

if ((det(A) == 0) || (det(B) == 0))
    fprintf('\n\nSome matrices are degenerate. End of work!\n\n')
    return;
end

invA = inv(A);
invB = inv(B);

p0 = [1 -1; 1 1];
p0 = vertcat(p0, -p0);
h0 = convhull(polyshape(p0,'Simplify',false));

[xlim,ylim] = boundingbox(h0);
scale0 = 1 / max(xlim(2), ylim(2));
h0 = scale(h0, scale0);

%% Computations
    fprintf('\n  #   rho_min    gamma    rho_max  Num_edges\n')


while true

    p0 = h0.Vertices;

    p1 = p0 * transpose(invA);
    p2 = p0 * transpose(invB);

    h1 = convhull(polyshape(p1,'Simplify',false));
    h2 = convhull(polyshape(p2,'Simplify',false));
    h12 = intersect(h1,h2);

    [rho_min,rho_max] = MinMax_normG_div_normH(h12, h0);
    gamma = (rho_max + rho_min) / 2;

    h00 = intersect(h0,scale(h12, gamma));
    h1 = scale(h1, gamma);
    h2 = scale(h2, gamma);

    [x00,y00] = boundingbox(h00);
    scale00 = 1 / max(x00(2), y00(2));
    h0 = scale(h00, scale00);

    iter = iter+1;
    [d,~] = size(h00.Vertices);
    fprintf('%3.0f. %.6f  %.6f  %.6f   %3.0f\n',iter,rho_min,gamma,rho_max,d)

    if (rho_max - rho_min) < tol
        break
    end
end

[x0,y0] = boundingbox(h0);
[x1,y1] = boundingbox(h1);
[x2,y2] = boundingbox(h2);
xyb = [x0(2), x1(2), x2(2), y0(2), y1(2), y2(2)];
bb = 1.1 * max(xyb);


plot(h0);
hold on
plot(h1, 'LineStyle','--','FaceColor','none');
hold on
plot(h2, 'LineStyle',':','FaceColor','none');
grid on
pbaspect([1 1 1]);
axis([-bb bb -bb bb]);

ss = get(0,'ScreenSize');
wx = 640;
wy = 485;
set(gcf, 'Position',  [ss(3)-wx-10, ss(4)-wy-90, wx, wy]);

sA = sprintf('$$A=\\left(\\begin{array}{rr}%4.2f&\\ %4.2f\\\\ %4.2f& %4.2f\\end{array}\\right)$$',A(1,1),A(1,2),A(2,1),A(2,2));
sB = sprintf('$$B=\\left(\\begin{array}{rr}%4.2f&\\ %4.2f\\\\ %4.2f& %4.2f\\end{array}\\right)$$',B(1,1),B(1,2),B(2,1),B(2,2));
sR = sprintf('$$\\rho=%8.6f$$',gamma);

title('A Barabanov norm for the matrix set \{A,B\}');
sMat=strcat('\qquad\qquad\qquad $$\|x\|=\frac{1}{\rho}\max\{\|Ax\|,\|Bx\|\}$$,\quad',sA,',\quad',sB,',\quad',sR);
xlabel(sMat,'Interpreter','latex','VerticalAlignment','top')

legend({'$$~~\|x\|=1$$','$$\|Ax\|=\rho$$','$$\|Bx\|=\rho$$'},...
    'Interpreter','latex','Location','BestOutside');


function n = pNorm(x, y, h)
    [xlim,ylim] = boundingbox(h);
    scale = 0.5 * sqrt(((xlim(2) - xlim(1))^2 + (ylim(2) - ylim(1))^2) /...
        (x^2 + y^2));
    ll = [0 0; scale*x scale*y];
    [in,~] = intersect(h,ll);
    n = sqrt((x^2 + y^2) / ((in(2,1)-in(1,1))^2 + (in(2,2)-in(1,2))^2));
end

function [fmin,fmax] = MinMax_normG_div_normH(g,h)
    pg = g.Vertices;
    [dimg,~] = size(pg);
    fmin = 1 / pNorm(pg(1,1), pg(1,2), h);
    fmax = fmin;
    for i = 2:dimg
        n = 1 / pNorm(pg(i,1), pg(i,2), h);
        fmin = min(fmin,n);
        fmax = max(fmax,n);
    end
    ph = h.Vertices;
    [dimh,~] = size(ph);
    for i = 1:dimh
        n = pNorm(ph(i,1), ph(i,2), g);
        fmin = min(fmin,n);
        fmax = max(fmax,n);
    end
end

