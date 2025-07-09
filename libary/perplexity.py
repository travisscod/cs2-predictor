import requests
import json
from typing import Dict, List, Any
from pydantic import BaseModel

class BettingRecommendation(BaseModel):
    match_id: str
    team_1: str
    team_2: str
    recommended_bet: str
    ev_percentage: float
    model_confidence: float
    bookmaker_odds: float
    stake_amount: float
    expected_return: float
    reasoning: str

class BettingPlan(BaseModel):
    bankroll_status: Dict[str, float]
    single_bets: List[BettingRecommendation]
    parlay_suggestion: Dict[str, Any]
    risk_assessment: Dict[str, str]
    adjustments_made: List[str]
    total_risk_percentage: float

class PerplexityBettingAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_betting_data(self, 
                           betting_prompt: str, 
                           matches_data: str, 
                           bankroll: float = 10) -> Dict[str, Any]:
        """
        Analyze betting data using Perplexity's deep research capabilities
        with structured JSON response
        """
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are an expert sports betting analyst and risk management specialist. 
                    Analyze the provided betting data and return ONLY a JSON response following the exact schema provided.
                    Current bankroll: ${bankroll}
                    
                    Your analysis should include:
                    - Expected Value calculations using EV% = (model_prob × odds) - 1
                    - Kelly Criterion stake sizing (use fractional Kelly at 25% for safety)
                    - Risk management keeping total daily risk under 20% of bankroll
                    - Cross-validation with current sports data and news
                    - Confidence thresholds based on model reliability
                    """
                },
                {
                    "role": "user", 
                    "content": f"{betting_prompt}\n\nMatches Data:\n{matches_data}"
                }
            ],
            "reasoning_effort": "high",
            "search_mode": "web",
            "return_related_questions": False,
            "web_search_options": {
                "search_context_size": "high"
            }
        }
        print(json.dumps(payload, indent=2))

        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Extract JSON response
            print(response.text)
            result = response.json()
            betting_analysis = json.loads(result["choices"][0]["message"]["content"])
            
            return {
                "success": True,
                "analysis": betting_analysis,
                "raw_response": result
            }
            
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON parsing error: {str(e)}"}


class AdvancedBettingScript:
    def __init__(self, api_key: str, initial_bankroll: float = 10.0):
        self.analyzer = PerplexityBettingAnalyzer(api_key)
        self.bankroll = initial_bankroll
        self.min_confidence_threshold = 0.65  # Minimum confidence for betting
        self.max_daily_risk = 0.20  # 20% max daily risk
        
    def calculate_ev_percentage(self, model_prob: float, odds: float) -> float:
        """Calculate Expected Value percentage"""
        return (model_prob * odds) - 1
    
    def kelly_stake_size(self, 
                        ev_percentage: float, 
                        model_prob: float, 
                        odds: float, 
                        bankroll: float,
                        kelly_fraction: float = 0.25) -> float:
        """
        Calculate optimal stake using fractional Kelly criterion
        Kelly% = (bp - q) / b
        Where: b = odds-1, p = probability, q = 1-p
        """
        if ev_percentage <= 0:
            return 0
            
        b = odds - 1
        p = model_prob
        q = 1 - p
        
        kelly_percentage = (b * p - q) / b
        fractional_kelly = kelly_percentage * kelly_fraction
        
        # Cap at reasonable limits
        stake = min(bankroll * fractional_kelly, bankroll * 0.05)  # Max 5% per bet
        return max(0, stake)
    
    def process_matches(self, matches_json: str, betting_prompt: str) -> Dict[str, Any]:
        """Process daily matches with comprehensive analysis"""
        
        enhanced_prompt = f"""
        {betting_prompt}
        
        ADDITIONAL REQUIREMENTS:
        
        RESEARCH VALIDATION:
        - Cross-check predictions with current team news, injuries, roster changes
        - Validate against recent team performance and head-to-head records  
        - Consider betting market sentiment and line movements
        - Factor in competition strength and opponent quality
        
        CONFIDENCE CALIBRATION:
        - Set confidence threshold at 65% minimum for betting recommendations
        - Higher confidence (>80%) allows for larger stake sizing
        - Lower confidence (65-80%) gets minimal stakes
        - Flag any predictions below 65% as "no bet"
        
        ADVANCED STAKE SIZING:
        - Use 25% fractional Kelly for optimal risk-adjusted growth
        - Cap individual bets at 5% of bankroll maximum
        - Ensure total daily exposure stays under 20% of bankroll
        - Provide specific dollar amounts based on current bankroll
        
        PARLAY STRATEGY:
        - Only suggest parlays with 3-5 legs maximum
        - All parlay legs must have 75%+ confidence
        - Parlay stake should be 2-3% of bankroll maximum
        - Calculate true parlay probability (multiply individual probabilities)
        
        OUTPUT FORMAT:
        Return structured JSON with exact calculations, reasoning, and actionable recommendations.
        Include backup options if main recommendations don't meet criteria.
        """
        
        result = self.analyzer.analyze_betting_data(
            betting_prompt=enhanced_prompt,
            matches_data=matches_json,
            bankroll=self.bankroll
        )
        
        if result["success"]:
            # Update bankroll tracking
            analysis = result["analysis"]
            if "bankroll_status" in analysis:
                self.bankroll = analysis["bankroll_status"].get("updated_bankroll", self.bankroll)
        
        return result
    
    def format_betting_plan(self, analysis: Dict[str, Any]) -> str:
        """Format the JSON response into readable betting plan"""
        if not analysis:
            return "No betting recommendations available."
            
        plan = []
        plan.append("DAILY BETTING PLAN")
        plan.append("=" * 50)
        
        # Bankroll status
        if "bankroll_status" in analysis:
            status = analysis["bankroll_status"]
            plan.append(f"Current Bankroll: ${status.get('current_bankroll', 'N/A')}")
            plan.append(f"Total Risk Today: {status.get('total_risk_percentage', 'N/A')}%")
            plan.append("")
        
        # Single bets
        if "single_bets" in analysis and analysis["single_bets"]:
            plan.append("RECOMMENDED SINGLE BETS:")
            for i, bet in enumerate(analysis["single_bets"], 1):
                plan.append(f"""
                {i}. {bet.get('team_1', 'N/A')} vs {bet.get('team_2', 'N/A')}
                   Bet: {bet.get('recommended_bet', 'N/A')}
                   EV: {bet.get('ev_percentage', 'N/A')}%
                   Confidence: {bet.get('model_confidence', 'N/A')}%
                   Odds: {bet.get('bookmaker_odds', 'N/A')}
                   Stake: ${bet.get('stake_amount', 'N/A')}
                   Expected Return: ${bet.get('expected_return', 'N/A')}
                   Reasoning: {bet.get('reasoning', 'N/A')}
                """)
        
        # Parlay suggestion
        if "parlay_suggestion" in analysis and analysis["parlay_suggestion"]:
            parlay = analysis["parlay_suggestion"]
            plan.append("\nPARLAY SUGGESTION:")
            plan.append(f"   Combined Odds: {parlay.get('combined_odds', 'N/A')}")
            plan.append(f"   Stake: ${parlay.get('stake_amount', 'N/A')}")
            plan.append(f"   Potential Payout: ${parlay.get('potential_payout', 'N/A')}")
            plan.append(f"   Risk Level: {parlay.get('risk_level', 'N/A')}")
        
        # Risk assessment
        if "risk_assessment" in analysis:
            risk = analysis["risk_assessment"]
            plan.append(f"\nRISK ASSESSMENT:")
            plan.append(f"   Overall Risk: {risk.get('overall_risk', 'N/A')}")
            plan.append(f"   Recommendation: {risk.get('recommendation', 'N/A')}")
        
        return "\n".join(plan)

# Usage Example
def main():
    # Initialize the betting script
    api_key = "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"
    betting_script = AdvancedBettingScript(api_key, initial_bankroll=10.0)
    
    # Your betting prompt (as provided)
    betting_prompt = """
    I will provide JSON files that contain predictions from my AI betting model. Each entry includes:
    
    Predicted win probabilities for both teams
    Model confidence score  
    Bookmaker odds for each team
    
    I currently have a $10 bankroll, and I want to maximize long-term profit by placing +EV bets with proper stake sizing and low risk.
    
    Please do the following:
    
    1. Calculate Expected Value (EV%)
    For each potential bet, calculate EV% using: EV% = (model_prob × odds) - 1
    
    2. Identify Profitable Bets  
    Filter and rank bets with positive EV%, starting with the highest. Recommend bets where both:
    EV% is positive
    Model confidence is high (suggest a threshold based on the data)
    
    3. Suggest Bet Types & Stake Sizing
    Based on my $10 bankroll:
    Recommend stake amounts for each bet (flat or proportional to confidence/EV%)
    Keep total risk per day under 20% of bankroll
    Suggest low-risk combo/parlay using 3–5 of the strongest picks, with a small stake (e.g., 10% or less of bankroll)
    
    4. Do Your Own Research
    Cross-check my AI's predictions using current data: injuries, roster changes, team form, live betting sentiment, and analyst insights
    If a team has inflated stats from playing weak opponents, adjust confidence using league/competition strength (e.g., HLTV rankings, Valve, Elo)
    
    5. Output a Final Betting Plan
    For today's matches:
    List single bets to place (EV%, confidence, stake, expected return)
    Suggest parlay if it makes sense
    Include any overrides or adjustments based on your own research
    
    Focus on actionable bets with real profit potential.
    """
    
    # Sample matches data (replace with your actual JSON data)
    matches_data = """
    {
        "matches": [
            {
                "match_id": "001",
                "team_1": "Team A",
                "team_2": "Team B", 
                "team_1_win_prob": 0.65,
                "team_2_win_prob": 0.35,
                "model_confidence": 0.78,
                "bookmaker_odds": {
                    "team_1": 1.85,
                    "team_2": 2.20
                }
            },
            {
                "match_id": "002", 
                "team_1": "Team C",
                "team_2": "Team D",
                "team_1_win_prob": 0.55,
                "team_2_win_prob": 0.45, 
                "model_confidence": 0.82,
                "bookmaker_odds": {
                    "team_1": 1.75,
                    "team_2": 2.15
                }
            }
        ]
    }
    """
    
    # Process the matches
    result = betting_script.process_matches(matches_data, betting_prompt)
    
    if result["success"]:
        # Print formatted betting plan
        formatted_plan = betting_script.format_betting_plan(result["analysis"])
        print(formatted_plan)
        
        # Return raw JSON for programmatic use
        return result["analysis"]
    else:
        print(f"Error: {result['error']}")
        return None

if __name__ == "__main__":
    betting_analysis = main()
