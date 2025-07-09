
# Calculate Expected Value (EV%)
def calculate_ev(probability, odds):
    return (probability * odds) - 1

# Calculate Kelly Criterion stake
def kelly_criterion(probability, odds, bankroll):
    b = odds - 1  # net odds
    q = 1 - probability
    fraction = (b * probability - q) / b
    return max(0, fraction) * bankroll

# Analyze match and generate betting recommendations
def analyze_match(data, bankroll):
    # Extract data
    prob1 = data['team1_win_probability']
    prob2 = data['team2_win_probability']
    odds1 = float(data['team1_odds'])
    odds2 = float(data['team2_odds'])
    confidence = data['confidence']
    
    # Calculate EVs
    ev1 = calculate_ev(prob1, odds1)
    ev2 = calculate_ev(prob2, odds2)
    
    # Determine profitable bets
    profitable_bets = []
    if ev1 > 0 and confidence in ['high', 'medium']:
        profitable_bets.append(('team1', ev1, odds1, prob1))
    if ev2 > 0 and confidence in ['high', 'medium']:
        profitable_bets.append(('team2', ev2, odds2, prob2))
    
    # Sort by highest EV
    profitable_bets.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate stakes using Kelly Criterion
    recommendations = []
    daily_risk_limit = bankroll * 0.20  # 20% of bankroll
    
    for team, ev, odds, prob in profitable_bets:
        full_kelly = kelly_criterion(prob, odds, bankroll)
        quarter_kelly = full_kelly * 0.25  # Conservative approach
        
        # Apply daily risk limit
        stake = min(quarter_kelly, daily_risk_limit)
        expected_return = stake * ev
        
        recommendations.append({
            'team': 'Nexus Gaming' if team == 'team1' else 'RUBY',
            'market': 'Match Winner',
            'odds': odds,
            'ev%': ev * 100,
            'confidence': confidence,
            'stake': stake,
            'expected_return': expected_return
        })
    
    return recommendations