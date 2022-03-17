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
avg_return_blue=[]
avg_return_purple=[]

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
plt.plot(eps,purple_returns, color="orange")
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.legend(['blue_team','purple_team'])
plt.savefig('team_reward.png')


plt.figure()
for i in range(1,n_eps+1):
    avg_elo_blue.append(np.mean(blue_ratings[i-min(i,200):i+1]))
    avg_elo_purple.append(np.mean(purple_ratings[i-min(i,200):i+1]))
plt.plot(eps,avg_elo_blue, color="blue")
plt.plot(eps,avg_elo_purple, color="orange")
plt.xlabel('episodes')
plt.ylabel('elo_ratings')
plt.legend(['blue_team','purple_team'])
plt.savefig('team_rating.png')

plt.figure()
team=['team_blue','team_purple']
plt.bar(team,[np.sum(blue_win),np.sum(purple_win)], color='blue')
plt.bar(team,[np.sum(1-np.array(blue_win)),np.sum(1-np.array(purple_win))],bottom=[np.sum(blue_win),np.sum(purple_win)], color='orange')
plt.legend(['win','loss'])
plt.savefig('win_loss.png')


plt.figure()
plt.plot(eps,returns, color="orange",lw=2, alpha=0.1, label="Lighten")
avg_returns_cal=[]
for i in range(1,n_eps+1):
    avg_returns_cal.append(np.mean(returns[i-min(i,100):i+1]))
plt.plot(eps,avg_returns_cal, color="orange")
plt.xlabel('episodes')
plt.ylabel('game_rewards')
plt.savefig('game_rewards.png')

plt.figure()
plt.plot(eps,noise, color="orange")
plt.xlabel('episodes')
plt.ylabel('noise_rate')
plt.savefig('noise_rate.png')

plt.figure()
plt.plot(eps, episode_len, color="orange",lw=2, alpha=0.1, label="Lighten")
for i in range(1,n_eps+1):
    avg_episode_len.append(np.mean(episode_len[i-min(i,100):i+1]))
plt.plot(eps,avg_episode_len, color="orange")
plt.xlabel('episodes')
plt.ylabel('episode_len')
plt.savefig('episode_len.png')


# actor_loss1=[]
# mean_actor_loss1=[]
# critic_loss1=[]
# mean_critic_loss1=[]
# actor_loss2=[]
# mean_actor_loss2=[]
# critic_loss2=[]
# mean_critic_loss2=[]
# with open('loss_for_agent0.txt', 'r') as datafile:
#     plotting = csv.reader(datafile)
    
#     for ROW in plotting:
#         actor_loss1.append(float(ROW[0]))
#         critic_loss1.append(float(ROW[1]))

# with open('loss_for_agent1.txt', 'r') as datafile:
#     plotting = csv.reader(datafile)
    
#     for ROW in plotting:
#         actor_loss2.append(float(ROW[0]))
#         critic_loss2.append(float(ROW[1]))

# n_eps = len(actor_loss1)
# eps = range(1,n_eps+1)
# plt.figure()
# for i in range(1,n_eps+1):
#     mean_actor_loss1.append(np.mean(actor_loss1[i-min(i,100):i+1]))
# plt.plot(eps,mean_actor_loss1, color="orange")
# plt.xlabel('episodes')
# plt.ylabel('actor_loss')
# plt.savefig('actor_loss1.png')

# plt.figure()
# for i in range(1,n_eps+1):
#     mean_critic_loss1.append(np.mean(critic_loss1[i-min(i,200):i+1]))
# plt.plot(eps,mean_critic_loss1, color="orange")
# plt.xlabel('episodes')
# plt.ylabel('critic_loss')
# plt.savefig('critic_loss1.png')

# plt.figure()
# for i in range(1,n_eps+1):
#     mean_actor_loss2.append(np.mean(actor_loss2[i-min(i,100):i+1]))
# plt.plot(eps,mean_actor_loss2, color="orange")
# plt.xlabel('episodes')
# plt.ylabel('actor_loss')
# plt.savefig('actor_loss2.png')

# plt.figure()
# for i in range(1,n_eps+1):
#     mean_critic_loss2.append(np.mean(critic_loss2[i-min(i,200):i+1]))
# plt.plot(eps,mean_critic_loss2, color="orange")
# plt.xlabel('episodes')
# plt.ylabel('critic_loss')
# plt.savefig('critic_loss2.png')