import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
purple_returns = []
blue_returns = []
purple_ratings = []
blue_ratings = []
purple_win = []
blue_win = []
returns=[]
avg_returns=[]
episode_len=[]
noise=[]
avg_episode_len=[]
avg_elo_blue=[]
avg_elo_purple=[]

with open('returns.txt', 'r') as datafile:
    plotting = csv.reader(datafile)
    
    for ROW in plotting:
        blue_returns.append(float(ROW[0]))
        purple_returns.append(float(ROW[1]))
        blue_ratings.append(float(ROW[2]))
        purple_ratings.append(float(ROW[3]))
        blue_win.append(int(ROW[4]))
        purple_win.append(int(ROW[5]))
        returns.append(float(ROW[6]))
        avg_returns.append(float(ROW[7]))
        noise.append(float(ROW[8]))
        episode_len.append(float(ROW[9]))


n_eps = len(blue_returns)
eps = range(1,n_eps+1)
 
plt.figure()
plt.plot(eps,blue_returns, color="blue")
plt.xlabel('episodes')
plt.ylabel('avg returns')
plt.legend(['blue team return'])
plt.savefig('blue_team_return.png')

plt.figure()
plt.plot(eps,purple_returns, color="purple")
plt.xlabel('episodes')
plt.ylabel('avg returns')
plt.legend(['purple team return'])
plt.savefig('purple_team_return.png')

plt.figure()
plt.plot(eps,blue_ratings, color="blue")
for i in range(1,n_eps+1):
    avg_elo_blue.append(np.mean(blue_ratings[i-min(i,100):i+1]))
plt.plot(eps,avg_elo_blue, color="orange")
plt.xlabel('episodes')
plt.ylabel('elo ratings')
plt.legend(['blue team rating'])
plt.savefig('blue_team_rating.png')

plt.figure()
plt.plot(eps,purple_ratings, color="purple")
for i in range(1,n_eps+1):
    avg_elo_purple.append(np.mean(purple_ratings[i-min(i,100):i+1]))
plt.plot(eps,avg_elo_purple, color="orange")
plt.xlabel('episodes')
plt.ylabel('elo ratings')
plt.legend(['purple team rating'])
plt.savefig('purple_team_rating.png')

plt.figure()
plt.step(eps,np.cumsum(blue_win), color="blue", where='mid')
plt.xlabel('episodes')
plt.ylabel('win')
plt.legend(['blue team win'])
plt.savefig('blue_team_win.png')

plt.figure()
plt.step(eps,np.cumsum(purple_win), color="purple", where='mid')
plt.xlabel('episodes')
plt.ylabel('win')
plt.legend(['purple team win'])
plt.savefig('purple_team_win.png')

plt.figure()
plt.plot(eps,returns, color="blue")
plt.plot(eps,avg_returns, color="orange")
plt.xlabel('episodes')
plt.ylabel('returns')
plt.legend(['returns','avg_returns'])
plt.savefig('returns.png')


plt.figure()
plt.plot(eps,noise, color="orange")
plt.xlabel('episodes')
plt.ylabel('noise_rate')
plt.savefig('noise_rate.png')

plt.figure()
plt.plot(eps, episode_len, color="blue")
for i in range(1,n_eps+1):
    avg_episode_len.append(np.mean(episode_len[i-min(i,100):i+1]))
plt.plot(eps,avg_episode_len, color="orange")
plt.xlabel('episodes')
plt.ylabel('episode_len')
plt.savefig('episode_len.png')


# actor_loss1=[]
# critic_loss1=[]
# actor_loss2=[]
# critic_loss2=[]
# with open('loss_for_agent0.txt', 'r') as datafile:
#     plotting = csv.reader(datafile)
    
#     for ROW in plotting:
#         actor_loss1.append(ROW[0].float().cpu().numpy())
#         #critic_loss1.append(float(ROW[1].cpu().numpy()))

# # with open('loss_for_agent1.txt', 'r') as datafile:
# #     plotting = csv.reader(datafile)
    
# #     for ROW in plotting:
# #         actor_loss2.append(float(ROW[0].cpu().numpy()))
# #         critic_loss2.append(float(ROW[1].cpu().numpy()))

# n_eps = len(actor_loss1)
# eps = range(1,n_eps+1)
# plt.figure()
# plt.plot(eps,actor_loss1, color="blue")
# plt.xlabel('episodes')
# plt.ylabel('actor_loss1')
# plt.legend(['actor_loss1'])
# plt.savefig('actor_loss1.png')
