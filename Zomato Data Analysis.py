#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sea
#%%
Zomato=pd.read_csv(r"C:\Users\lenovo\Desktop\ML PROJECT\zomato.csv")
print(Zomato)
#%%
Zomato.info()
#%%
Zomato.describe()
#%%
##Restaurants chains of Zomato
plt.figure(figsize=(10,10))
chains=Zomato['name'].value_counts()[:30]
sea.barplot(x=chains,y=chains.index,palette='deep')
plt.title("restaurants chains in Zomato",weight ='bold')
plt.xlabel("Number of outlets")
#%%
##Table Booking
x=Zomato['book_table'].value_counts()
c=['r','g']
fig, ax2 = plt.subplots()
ax2.pie(x,labels=x,colors=c,autopct='%1.01f%%',frame='True',shadow=5.2)
ax2.axis('equal')
plt.title("Table booking",weight ='bold')
plt.legend(x.index)
plt.show()

#%%
#How many restuarants do not accept online orders? 
##Accepting vs Not Accepting Online Orders
x=Zomato['online_order'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(x,labels=x,autopct='%1.1f%%', shadow=6,startangle=90,frame='True')
ax1.axis('equal')
plt.title("Accepting vs Not Accepting Online Orders",weight ='bold')
plt.legend(x.index)
plt.show()
#%%
#Rating distribution
plt.figure(figsize=(20,10))
rating=Zomato['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()
sea.distplot(rating,bins=20)
plt.title("Rating distribution ",weight ='bold')
#%%
#Cost Vs Rating
cost_dist=Zomato[['rate','approx_cost(for two people)','location','name','rest_type']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
plt.figure(figsize=(10,7))
sea.scatterplot(x="rate",y='approx_cost(for two people)',data=cost_dist)
plt.title("Rating",weight ='bold')
plt.show()
#%%
plt.figure(figsize=(20,10))
Zomato['online_order']
sea.scatterplot(x="rate",y='approx_cost(for two people)',hue=Zomato['online_order'],data=cost_dist)
plt.title("Cost Vs Rating",weight ='bold')
#%%
#Distribution of cost for two people
plt.figure(figsize=(20,10))
sea.distplot(cost_dist['approx_cost(for two people)'],color='r')
plt.title(" Distribution of cost for two people ",weight ='bold')
plt.show()
#%%
#Location And Cuisines Data.
df_1=Zomato.groupby(['location','cuisines']).agg('count')
data=df_1.sort_values(['url'],ascending=False).groupby(['location'],as_index=False).apply(lambda x : x.sort_values(by="url",ascending=False).head(3))['url'].reset_index().rename(columns={'url':'count'})
data.head(10)
#%%
cost_dist=Zomato[['rate','approx_cost(for two people)','location','name','rest_type']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
def return_budget(location,rest):
    budget=cost_dist[(cost_dist['approx_cost(for two people)']<=400) & (cost_dist['location']==location) & 
                     (cost_dist['rate']>4) & (cost_dist['rest_type']==rest)]
    return(budget['name'].unique())
return_budget('BTM',"Quick Bites")
#%%
#Which are the foodie areas? 
plt.figure(figsize=(20,10))
Rest_locations=Zomato['location'].value_counts()[:20]
plt.title("Which are the food areas",weight ='bold')
sea.barplot(Rest_locations,Rest_locations.index,palette="summer")
#%%
#Which are the most popular cuisines of Zomato? 
plt.figure(figsize=(20,10))
cuisines=Zomato['cuisines'].value_counts()[:10]
sea.barplot(cuisines,cuisines.index)
plt.xlabel('Count')
plt.title("Most popular cuisines of Zomato")
#%%
#Is there any difference b/w votes of restaurants accepting and not accepting online orders? 
plt.figure(figsize=(10,7))
votes_yes=Zomato[Zomato['online_order']=="Yes"]['votes']
votes_no=Zomato[Zomato['online_order']=="No"]['votes']
sea.boxplot(votes_yes,orient='v',color = 'r',width=800)
plt.xlabel('accepting online orders')
plt.title("Box Plots of votes_yes")
#%%
sea.boxplot(votes_no,orient='v',color = 'b',width=800)
plt.xlabel('Not accepting online orders')
plt.title("Box Plots of votes_No")
#%%
#Which are the most common restaurant type in Zomato? 
plt.figure(figsize=(15,10))
rest=Zomato['rest_type'].value_counts()[:20]
sea.barplot(rest,rest.index)
plt.title("Restaurant types")
plt.xlabel("count")
#%%