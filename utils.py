import matplotlib.pyplot as plt

def plot_solution(sol,y_exact,title):
    fig,ax = plt.subplots(figsize=(10,8))

    ax.plot(sol[0][0],sol[0][1],'k+', label="h = 0.1") #,'k--',dashes=[10,8]
    ax.plot(sol[1][0],sol[1][1],'k--',dashes=[10,8], label="h = 0.01") #,'k--',dashes=[10,8]
    ax.plot(sol[2][0],sol[2][1],'k--', label="h = 0.001") #,'k--',dashes=[10,8]
    ax.plot(y_exact[0],y_exact[1],'k-', label="Solução Exata")
    ax.set_title(title,fontsize=18, pad = 15)
    ax.set_xlabel('$y_1$(t)', fontsize=15, labelpad = 15)
    ax.set_ylabel('$y_2$(t)', fontsize=15, rotation = 0, labelpad = 10)
    ax.tick_params(axis='both', which='major', labelsize=15, pad=15)
    ax.legend(fontsize=15,bbox_to_anchor=(1.01, 0.65), loc=2)