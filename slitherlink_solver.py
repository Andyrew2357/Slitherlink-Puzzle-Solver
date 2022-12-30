#------------------------------------------------------------------------------------------------------------------
# IMPORT PACKAGES

import numpy as np
from matplotlib import pyplot as plt
import itertools as it
import time

#------------------------------------------------------------------------------------------------------------------
# INITIALIZE THE CELLS, EDGES, & STACK

# Cells
with open("40_30_easy.txt","r") as f:
    dimensions=f.readline()
    m,n=(int(s) for s in dimensions.split(","))
    cells=-1*np.ones(shape=(m,n),dtype=int)
    b=f.readline()
    i=0
    for k in b:
        r,c=i//n,i%n
        if k.isdigit():
            cells[r,c]=int(k)
            i+=1
        else:
            i+=ord(k)-96
m,n=cells.shape

# Edges
# The third dimension of edges is ordered 'right', 'up', 'left', 'down'
edges=np.zeros((m+1,n+1,4),dtype=int)

for row in range(m+1):
    edges[row,-1,0]-=1
    edges[row,0,2]-=1
for col in range(n+1):
    edges[0,col,1]-=1
    edges[-1,col,3]-=1

# Stack initialized with the largest size it could possibly need
stack=np.zeros((2*m*n+m+n,3),dtype=int)-1
cur_stack_size=0

# Adjacency Relations
dr=[0,-1,0,1]
dc=[1,0,-1,0]

#------------------------------------------------------------------------------------------------------------------
# FUNCTIONS FOR MANIPULATING AND CHECKING THE BOARD

def print_solution():
    # Add the vertices.
    plt.gca().invert_yaxis()
    r,c=np.meshgrid(np.arange(m+1),np.arange(n+1))
    r=r.reshape(1,(n+1)*(m+1))[0]
    c=c.reshape(1,(n+1)*(m+1))[0]
    fig=plt.scatter(c,r,color="k",marker="s",s=40/min(m,n))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.set_aspect(1)
    for row in range(m+1):
        for col in range(n+1):
            for edge in range(4):
                if edges[row,col,edge]==1:
                    plt.plot([col,col+dc[edge]],[row,row+dr[edge]],color="k",linewidth=20/min(m,n))
                elif edges[row,col,edge]==-1:
                    plt.text(col+0.5*dc[edge],row+0.5*dr[edge],"x",fontsize=int(100/min(m,n)),
                    bbox={'facecolor':'none','alpha':1,'edgecolor':'none'},
                    ha='center', va='center')
    for row in range(m):
        for col in range(n):
            if cells[row,col] == -1:
                continue
            plt.text(col+0.5,row+0.5,str(cells[row,col]),fontsize=int(150/min(m,n)),
                    bbox={'facecolor':'none','alpha':1,'edgecolor':'none','pad':1},
                    ha='center', va='center')
    plt.show()

def cells_satisfied():
    # Check to see if all cells are satisfied.
    # If they are, the solution has been found.
    for row in range(m):
        for col in range(n):
            if cells[row,col] == -1:
                continue
            num_line=0
            for i in [0,3]:
                if edges[row,col,i] == 1:
                    num_line+=1
            for i in [1,2]:
                if edges[row+1,col+1,i] == 1:
                    num_line+=1
            if not num_line == cells[row,col]:
                return False
    print("Solve Time: ", round(time.perf_counter()-start_time,4))
    global counter
    global counterl
    print("backtracking calls: ", counter)
    print("logic calls: ", counterl)
    print_solution()
    quit()

def count_fillings(row,col):
    if row < 0 or col < 0 or row >= m or col >= n:
        return True
    if cells[row,col] == -1:
        return True
    num_line=0
    num_cross=0
    for i in [0,3]:
        if edges[row,col,i] == 1:
            num_line+=1
        elif edges[row,col,i] == -1:
            num_cross+=1
    for i in [1,2]:
        if edges[row+1,col+1,i] == 1:
            num_line+=1
        elif edges[row+1,col+1,i] == -1:
            num_cross+=1
    
    return not (num_line > cells[row,col] or num_cross > 4 - cells[row,col])

def check_board(row,col,edge):

    # Check for vertex-wise contradictions
    num_line=0
    num_cross=0
    for e_ in range(4):
        if edges[row,col,e_] == 1:
            num_line+=1
        elif edges[row,col,e_] == -1:
            num_cross+=1
    
    if num_line > 2:
        return False
    if num_line == 1 and num_cross == 3:
        return False

    num_line=0
    num_cross=0
    for e_ in range(4):
        if edges[row+dr[edge],col+dc[edge],e_] == 1:
            num_line+=1
        elif edges[row+dr[edge],col+dc[edge],e_] == -1:
            num_cross+=1
    
    if num_line > 2:
        return False
    if num_line == 1 and num_cross == 3:
        return False

    # Check for cell-wise contradictions
    if edge == 0 or edge == 1:
        if not count_fillings(row-1,col):
            return False
    if edge == 1 or edge == 2:
        if not count_fillings(row-1,col-1):
            return False
    if edge == 2 or edge == 3:
        if not count_fillings(row,col-1):
            return False
    if edge == 3 or edge == 0:
        if not count_fillings(row,col):
            return False

    # Check for board-wise contradictions
    # Determine if the current segment forms a loop and of what size.
    if edges[row,col,edge] == -1:
        return True

    for i in range(5):
        if i == 4:
            return True
        if i == edge:
            continue
        if edges[row,col,i] == 1:
            d=i
            break
    
    prev_d=(d+2)%4
    cur_row,cur_col=row+dr[d],col+dc[d]
    loop_len=2    

    while True:
        for i in range(5):
            if i == 4:
                return True
            if i == prev_d:
                continue
            if edges[cur_row,cur_col,i] == 1:
                d=i
                break
        prev_d=(d+2)%4
        cur_row+=dr[d]
        cur_col+=dc[d]
        if (cur_row,cur_col)==(row,col):
            break
        else:
            loop_len+=1

    total_lines=0
    for r in range(m+1):
        for c in range(n+1):
            for e in [0,3]:
                if edges[r,c,e]==1:
                    total_lines+=1
    
    if loop_len < total_lines:
        return False
    else:
        return cells_satisfied()
    
def fill_edge(row,col,edge,state):
    # Check if the edge is already filled with the opposite state.
    # If it is, throw a contradiction. Then check if it already has
    # the specified state. If this is the case do nothing and return.

    if edges[row,col,edge] == -state:
        return False
    elif edges[row,col,edge] == state:
        return True
    
    # Fill in the edge for both rlevant vertices and add to stack.
    edges[row,col,edge]=state
    edges[row+dr[edge],col+dc[edge],(edge+2)%4]=state
    global cur_stack_size
    stack[cur_stack_size,:]=[row,col,edge]
    cur_stack_size+=1

    return check_board(row,col,edge)

def revert(k):
    global cur_stack_size
    for entry in range(k,cur_stack_size):
        r,c,e=stack[entry,0],stack[entry,1],stack[entry,2]
        edges[r,c,e]=0
        edges[r+dr[e],c+dc[e],(e+2)%4]=0
        for i in range(3):
            stack[entry,i]=-1
    cur_stack_size=k

#------------------------------------------------------------------------------------------------------------------
# SOLVE THE PUZZLE

def solve_corners():
    r=[-1,-1,0,0]
    c=[0,-1,-1,0]
    for i in range(4):
        if cells[r[i],c[i]] == 1:
            solve_logically(r[i],c[i],i,-1)
            solve_logically(r[i]+dr[i],c[i]+dc[i],(i+2)%4,-1)
            solve_logically(r[i],c[i],(i+1)%4,-1)
            solve_logically(r[i]+dr[(i+1)%4],c[i]+dc[(i+1)%4],(i+3)%4,-1)
        elif cells[r[i],c[i]] == 2:
            solve_logically(r[i]+dr[i],c[i]+dc[i],i,1)
            solve_logically(r[i]+2*dr[i],c[i]+2*dc[i],(i+2)%4,1)
            solve_logically(r[i]+dr[(i+1)%4],c[i]+dc[(i+1)%4],(i+1)%4,1)
            solve_logically(r[i]+2*dr[(i+1)%4],c[i]+2*dc[(i+1)%4],(i+3)%4,1)
        elif cells[r[i],c[i]] == 3:
            solve_logically(r[i],c[i],i,1)
            solve_logically(r[i]+dr[i],c[i]+dc[i],(i+2)%4,1)
            solve_logically(r[i],c[i],(i+1)%4,1)
            solve_logically(r[i]+dr[(i+1)%4],c[i]+dc[(i+1)%4],(i+3)%4,1)

def fill_zeros():
    for row in range(m):
        for col in range(n):
            if not cells[row,col] == 0:
                continue
            solve_logically(row,col,0,-1)
            solve_logically(row,col,3,-1)
            solve_logically(row+1,col,0,-1)
            solve_logically(row+1,col,1,-1)
            solve_logically(row,col+1,2,-1)
            solve_logically(row,col+1,3,-1)
            solve_logically(row+1,col+1,1,-1)
            edges[row+1,col+1,2]=-1

def precondition_board():
    # Perform a baord-wide search for patterns it knows how to fill in
    # prior to calling the backtracking function, ie.:
    #
    #     x           x               x
    # +   +   +     x +---+   +     x + x +
    # | 3 | 3 |       | 3             x 1             x
    # +   +   +       +   +   +       +   +       +   +---
    #     x                 3 |                     2 
    #                 +   +---+ x              ---+   + x
    #                         x                   x   x
    #
    # When iterating through the board and adding these, call on the
    # logical solver to fill in secondary edges. Then clear stack

    for row in range(m):
        for col in range(n):
            if cells[row,col] == 3:
                for d in range(4):
                    r,c=row+dr[d],col+dc[d]
                    if r<0 or c<0 or r>=m or c>=n:
                        continue
                    if cells[r,c] == 3:
                        dir1=3*((d+1)%2)
                        dir2=d%2
                        if d == 0:
                            r_,c_=row,col+1
                        elif d == 1:
                            r_,c_=row,col
                        elif d == 2:
                            r_,c_=row,col
                        else:
                            r_,c_=row+1,col
                        solve_logically(r_,c_,dir1,1)
                        solve_logically(r_-dr[dir2],c_-dc[dir2],dir1,1)
                        solve_logically(r_+dr[dir2],c_+dc[dir2],dir1,1)
                        if edges[r_-dr[dir1],c_-dc[dir1],dir1] == 0:
                            solve_logically(r_-dr[dir1],c_-dc[dir1],dir1,-1)
                        if edges[r_+dr[dir1],c_+dc[dir1],dir1] == 0:
                            solve_logically(r_+dr[dir1],c_+dc[dir1],dir1,-1)
                        break
                for k,r_,c_ in [(0,1,1),(1,-1,1),(2,-1,-1),(3,1,-1)]:
                    r,c=row+r_,col+c_
                    if r<0 or c<0 or r>=m or c>=n:
                        continue
                    if cells[r,c] == 3:
                        r1,c1=row,col
                        r2,c2=r,c
                        if c_>0:
                            c2+=1
                        else:
                            c1+=1
                        if r_>0:
                            r2+=1
                        else:
                            r1+=1
                        solve_logically(r1,c1,(k-1)%4,1)
                        solve_logically(r1,c1,k,1)
                        solve_logically(r2,c2,(k+1)%4,1)
                        solve_logically(r2,c2,(k+2)%4,1)
                        if edges[r1,c1,(k+1)%4] == 0:
                            solve_logically(r1,c1,(k+1)%4,-1)
                        if edges[r1,c1,(k+2)%4] == 0:
                            solve_logically(r1,c1,(k+2)%4,-1)
                        if edges[r2,c2,(k-1)%4] == 0:
                            solve_logically(r2,c2,(k-1)%4,-1)
                        if edges[r2,c2,k] == 0:
                            solve_logically(r2,c2,k,-1)
                        break

    global cur_stack_size
    for k in range(cur_stack_size):
        for i in range(3):
            stack[k,i]=-1

def cell_logic(row, col, state):
    if row < 0 or col < 0 or row >= m or col >= n:
        return True
    if cells[row,col] == -1:
        return True
    num_match = 0
    for r,c,e in [(row,col,0),(row,col,3),(row+1,col+1,1),(row+1,col+1,2)]:
        if edges[r,c,e] == state:
            num_match+=1
    if state == 1:
        if num_match == cells[row,col]:
            for r,c,e in [(row,col,0),(row,col,3),(row+1,col+1,1),(row+1,col+1,2)]:
                if edges[r,c,e] == 0:
                    if not solve_logically(r,c,e,-1): return False
    else:
        if num_match == 4-cells[row,col]:
            for r,c,e in [(row,col,0),(row,col,3),(row+1,col+1,1),(row+1,col+1,2)]:
                if edges[r,c,e] == 0:
                    if not solve_logically(r,c,e,1): return False
    return True

def three_logic(r,c,e):
    if e == 0 or e == 1:
        if 0<=r<m and 0<=c-1<n and cells[r,c-1] == 3:
            if not edges[r+1,c-1,0] == 1:
                if not solve_logically(r+1,c-1,0,1): return False
            if not edges[r+1,c-1,1] == 1:
                if not solve_logically(r+1,c-1,1,1): return False

            if e == 0 and not edges[r,c,1] == -1:
                if not solve_logically(r,c,1,-1): return False
            elif e == 1 and not edges[r,c,0] == -1:
                if not solve_logically(r,c,0,-1): return False
    if e == 1 or e == 2:
        if 0<=r<m and 0<=c<n and cells[r,c] == 3:
            if not edges[r+1,c+1,1] == 1:
                if not solve_logically(r+1,c+1,1,1): return False
            if not edges[r+1,c+1,2] == 1:
                if not solve_logically(r+1,c+1,2,1): return False

            if e == 1 and not edges[r,c,2] == -1:
                if not solve_logically(r,c,2,-1): return False
            elif e == 2 and not edges[r,c,1] == -1:
                if not solve_logically(r,c,1,-1): return False
    if e == 2 or e == 3:
        if 0<=r-1<m and 0<=c<n and cells[r-1,c] == 3:
            if not edges[r-1,c+1,2] == 1:
                if not solve_logically(r-1,c+1,2,1): return False
            if not edges[r-1,c+1,3] == 1:
                if not solve_logically(r-1,c+1,3,1): return False
            
            if e == 2 and not edges[r,c,3] == -1:
                if not solve_logically(r,c,3,-1): return False
            elif e == 3 and not edges[r,c,2] == -1:
                if not solve_logically(r,c,2,-1): return False
    if e == 3 or e == 0:
        if 0<=r-1<m and 0<=c-1<n and cells[r-1,c-1] == 3:
            if not edges[r-1,c-1,0] == 1:
                if not solve_logically(r-1,c-1,0,1): return False
            if not edges[r-1,c-1,3] == 1:
                if not solve_logically(r-1,c-1,3,1): return False
            
            if e == 0 and not edges[r,c,3] == -1:
                if not solve_logically(r,c,3,-1): return False
            elif e == 3 and not edges[r,c,0] == -1:
                if not solve_logically(r,c,0,-1): return False

    return True

def two_logic(r,c,e):
    if e == 0 or e == 1:
        if 0<=r<m and 0<=c-1<n and cells[r,c-1] == 2:
            b=False
            if edges[r+1,c-1,1] == -1 and not edges[r+1,c-1,0] == 1:
                if not solve_logically(r+1,c-1,0,1): return False
                b=True
            if edges[r+1,c-1,0] == -1 and not edges[r+1,c-1,1] == 1:
                b=True
                if not solve_logically(r+1,c-1,1,1): return False
            if b:
                if e == 0 and not edges[r,c,1] == -1:
                    if not solve_logically(r,c,1,-1): return False
                elif e == 1 and not edges[r,c,0] == -1:
                    if not solve_logically(r,c,0,-1): return False
    if e == 1 or e == 2:
        if 0<=r<m and 0<=c<n and cells[r,c] == 2:
            b=False
            if edges[r+1,c+1,2] == -1 and not edges[r+1,c+1,1] == 1:
                if not solve_logically(r+1,c+1,1,1): return False
                b=True
            if edges[r+1,c+1,1] == -1 and not edges[r+1,c+1,2] == 1:
                if not solve_logically(r+1,c+1,2,1): return False
                b=True
            if b:
                if e == 1 and not edges[r,c,2] == -1:
                    if not solve_logically(r,c,2,-1): return False
                elif e == 2 and not edges[r,c,1] == -1:
                    if not solve_logically(r,c,1,-1): return False
    if e == 2 or e == 3:
        if 0<=r-1<m and 0<=c<n and cells[r-1,c] == 2:
            b=False
            if edges[r-1,c+1,3] == -1 and not edges[r-1,c+1,2] == 1:
                if not solve_logically(r-1,c+1,2,1): return False
                b=True
            if edges[r-1,c+1,2] == -1 and not edges[r-1,c+1,3] == 1:
                if not solve_logically(r-1,c+1,3,1): return False
                b=True
            if b:            
                if e == 2 and not edges[r,c,3] == -1:
                    if not solve_logically(r,c,3,-1): return False
                elif e == 3 and not edges[r,c,2] == -1:
                    if not solve_logically(r,c,2,-1): return False
    if e == 3 or e == 0:
        if 0<=r-1<m and 0<=c-1<n and cells[r-1,c-1] == 2:
            b=False
            if edges[r-1,c-1,3] == -1 and not edges[r-1,c-1,0] == 1:
                if not solve_logically(r-1,c-1,0,1): return False
                b=True
            if edges[r-1,c-1,0] == -1 and not edges[r-1,c-1,3] == 1:
                if not solve_logically(r-1,c-1,3,1): return False
                b=True
            if b:
                if e == 0 and not edges[r,c,3] == -1:
                    if not solve_logically(r,c,3,-1): return False
                elif e == 3 and not edges[r,c,0] == -1:
                    if not solve_logically(r,c,0,-1): return False

    return True

def one_logic(r,c,edge1,edge2):
    e1,e2=(edge1,edge2) if edge1<edge2 else (edge2,edge1)
    
    if e1 == 0 and e2 == 1:
        if 0<=r<m and 0<=c-1<n and cells[r,c-1] == 1:
            if not edges[r+1,c-1,e1] == -1:
                if not solve_logically(r+1,c-1,e1,-1): return False
            if not edges[r+1,c-1,e2] == -1:
                if not solve_logically(r+1,c-1,e2,-1): return False
    elif e1== 1 and e2 == 2:
        if 0<=r<m and 0<=c<n and cells[r,c] == 1:
            if not edges[r+1,c+1,e1] == -1:
                if not solve_logically(r+1,c+1,e1,-1): return False
            if not edges[r+1,c+1,e2] == -1:
                if not solve_logically(r+1,c+1,e2,-1): return False
    elif e1== 2 and e2 == 3:
        if 0<=r-1<m and 0<=c<n and cells[r-1,c] == 1:
            if not edges[r-1,c+1,e1] == -1:
                if not solve_logically(r-1,c+1,e1,-1): return False
            if not edges[r-1,c+1,e2] == -1:
                if not solve_logically(r-1,c+1,e2,-1): return False
    else:
        if 0<=r-1<m and 0<=c-1<n and cells[r-1,c-1] == 1:
            if not edges[r-1,c-1,e1] == -1:
                if not solve_logically(r-1,c-1,e1,-1): return False
            if not edges[r-1,c-1,e2] == -1:
                if not solve_logically(r-1,c-1,e2,-1): return False
    
    return True

counterl=0
def solve_logically(row,col,edge,state):
    global counterl
    counterl+=1

    if not fill_edge(row,col,edge,state):
        return False
    
    # VERTEX-WISE LOGIC
    # If any of the involved vertices have two edges, fill in the others with crosses.
    # Alternatively, if there is a vertex with three crosses, fill the last one. ie.:
    #    |          (x)         x
    # (x)+(x)     (x)+--     (x)+ x
    #    |           |          x
    
    if state == 1:
        # vertex-wise logic given a line
        num_line=0
        num_cross=0
        for e_ in range(4):
            if edges[row,col,e_] == 1:
                num_line+=1
                if num_line == 2:
                    for e in range(4):
                        if edges[row,col,e] == 0:
                            if not solve_logically(row,col,e,-1): return False
                    break
            elif edges[row,col,e_] == -1:
                num_cross+=1
                if num_cross == 2:
                    for e in range(4):
                        if edges[row,col,e] == 0:
                            if not solve_logically(row,col,e,1): return False
                    break

        num_line=0
        num_cross=0
        for e_ in range(4):
            if edges[row+dr[edge],col+dc[edge],e_] == 1:
                num_line+=1
                if num_line == 2:
                    for e in range(4):
                        if edges[row+dr[edge],col+dc[edge],e] == 0:
                            if not solve_logically(row+dr[edge],col+dc[edge],e,-1): return False
                    break
            elif edges[row+dr[edge],col+dc[edge],e_] == -1:
                num_cross+=1
                if num_cross == 2:
                    for e in range(4):
                        if edges[row+dr[edge],col+dc[edge],e] == 0:
                            if not solve_logically(row+dr[edge],col+dc[edge],e,1): return False
                    break

    else:
        # vertex-wise logic given a cross
        num_line=0
        num_cross=0
        for e_ in range(4):
            if edges[row,col,e_] == -1:
                num_cross+=1
                if num_cross == 3:
                    for e in range(4):
                        if not edges[row,col,e] == -1:
                            if not solve_logically(row,col,e,-1): return False
            if edges[row,col,e_] == 1:
                num_line+=1
            if num_cross == 2 and num_line == 1:
                for e in range(4):
                    if edges[row,col,e] == 0:
                        if not solve_logically(row,col,e,1): return False
        
        num_line=0
        num_cross = 0
        for e_ in range(4):
            if edges[row+dr[edge],col+dc[edge],e_] == -1:
                num_cross+=1
                if num_cross == 3:
                    for e in range(4):
                        if not edges[row+dr[edge],col+dc[edge],e] == -1:
                            if not solve_logically(row+dr[edge],col+dc[edge],e,-1): return False
            if edges[row+dr[edge],col+dc[edge],e_] == 1:
                num_line+=1
            if num_cross == 2 and num_line == 1:
                for e in range(4):
                    if edges[row+dr[edge],col+dc[edge],e] == 0:
                        if not solve_logically(row+dr[edge],col+dc[edge],e,1): return False

    # CELL-WISE LOGIC
    # If a sufficient number of lines or crosses are filled, fill in the rest of the
    # edges surrounding the cell. ie:
    #   +---+       + x +      +(x)+
    #   | 2(x)     (|)3(|)     | 1(x)
    #   +(x)+       +---+      +(x)+

    if edge == 0 or edge == 1:
        if not cell_logic(row-1,col,state): return False
    if edge == 1 or edge == 2:
        if not cell_logic(row-1,col-1,state): return False
    if edge == 2 or edge == 3:
        if not cell_logic(row,col-1,state): return False
    if edge == 3 or edge == 0:
        if not cell_logic(row,col,state): return False

    # PATTERN-WISE LOGIC
    # There are a variety of recognizable patterns which allow us to fill in edges. ie.:
    #   x                           x
    # x + x +                     x +---+
    #   x 1             x           | 3            |
    #   +   +       +   +---        +   + x     ---+ x +
    #                 2                 |          x 2 | 
    #            ---+   + x                        +---+ x
    #               x   x                              x

    # Rules for cells with and 3 when there are two adjacent crosses
    
    if state == -1 and edges[row,col,(edge+1)%4] == -1:
        r,c=row-edge//2,col-((edge-1)%4)//2
        if 0 <= r < m and 0 <= c < n: 
            if cells[r,c] == 1 and not edges[row,col,(edge+2)%4] == -1:
                if not solve_logically(row,col,(edge+2)%4,-1): return False
            elif cells[r,c] == 3 and not edges[row,col,(edge+2)%4] == 1:
                if not solve_logically(row,col,(edge+2)%4,1): return False

    if state == -1 and edges[row,col,(edge-1)%4] == -1:
        r,c=row-((edge-1)%4)//2,col-((edge-2)%4)//2
        if 0 <= r < m and 0 <= c < n:
            if cells[r,c] == 1 and not edges[row,col,(edge+2)%4] == -1:
                if not solve_logically(row,col,(edge+2)%4,-1): return False
            elif cells[r,c] == 3 and not edges[row,col,(edge+2)%4] == 1:
                if not solve_logically(row,col,(edge+2)%4,1): return False

    # Rules for cells with 3 if theree is a line pointing in
    
    if state == 1:
        if not three_logic(row,col,edge): return False
        if not three_logic(row+dr[edge],col+dc[edge],(edge+2)%4): return False

    # Rules for cells with 1 if there are a line and a cross pointing in

    if state == 1:
        if edges[row,col,(edge+1)%4] == -1:
            if not one_logic(row,col,edge,(edge+1)%4): return False
        if edges[row,col,(edge-1)%4] == -1:
            if not one_logic(row,col,edge,(edge-1)%4): return False
        
        if edges[row+dr[edge],col+dc[edge],(edge+3)%4] == -1:
            if not one_logic(row+dr[edge],col+dc[edge],(edge+2)%4,(edge+3)%4): return False
        if edges[row+dr[edge],col+dc[edge],(edge+1)%4] == -1:
            if not one_logic(row,col,(edge+2)%4,(edge+1)%4): return False
    
    # For some reason, this isn't getting as much mileage
    # as I would expect, so I'll have to check for bugs.
    else:
        if edges[row,col,(edge+1)%4] == 1:
            if not one_logic(row,col,edge,(edge+1)%4): return False
        if edges[row,col,(edge-1)%4] == 1:
            if not one_logic(row,col,edge,(edge-1)%4): return False

    # Rules for cells with two with a line pointing in opposite to a cross
    if state == 1:
        if not two_logic(row,col,edge): return False
        if not two_logic(row+dr[edge],col+dc[edge],(edge+2)%4): return False
                
    # If we exhaust all logical methods without finding a
    # contradiction, return True and resort to backtracking
    return True

def find_next_open():
    for r in range(m+1):
        for c in range(n+1):
            for e in [0,3]:
                if edges[r,c,e] == 0:
                    return r,c,e

counter=0
def backtracking():
    global counter
    counter+=1
    if counter%10_000==0:
        print(counter,", ",counterl)
    # Take note of the current length of the stack.
    stack_len = cur_stack_size
    # Find the next open edge and attempt to put a line there.
    try:
        row,col,edge=find_next_open()
    except:
        return True

    if not solve_logically(row,col,edge,1):
        revert(stack_len)
        if not solve_logically(row,col,edge,-1): return False
        return backtracking()
    else:
        if not backtracking():
            revert(stack_len)
            if not solve_logically(row,col,edge,-1): return False
            return backtracking()            

start_time = time.perf_counter()
fill_zeros()
solve_corners()
precondition_board()
backtracking()

