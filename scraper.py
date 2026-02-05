import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()

DATA_DIR = SCRIPT_DIR / 'data'
INPUT_FILE = DATA_DIR / 'eu_consultation_responses.json'

async def scrape_consultations():
    base_url = 'https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/14671-On-farm-animal-welfare-for-certain-animals-modernisation-of-EU-legislation/feedback_en?p_id=19848&page='
    
    all_responses = []
    
    save_interval = 10  # Save every 10 pages
    max_retries = 3
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        page.set_default_timeout(15000)  # Increase timeout to 15 seconds
        
        for page_num in range(1, 69):
            url = f'{base_url}{page_num}'
            print(f'Scraping page {page_num}: {url}')
            
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    await page.goto(url, wait_until='networkidle', timeout=15000)
                    
                    # Try to wait for selector with shorter timeout
                    try:
                        await page.wait_for_selector('p.feedback-item-paragraph', timeout=5000)
                    except Exception:
                        # If selector not found, try parsing anyway
                        print(f'  ⚠ Selector timeout on page {page_num}, parsing anyway...')
                    
                    soup = BeautifulSoup(await page.content(), 'html.parser')
                    feedback_items = soup.find_all('p', class_='feedback-item-paragraph')
                    
                    if not feedback_items:
                        print(f'  ⚠ No items found on page {page_num} - reached end or page is empty')
                        # Continue to next page instead of breaking
                        success = True
                        continue
                    
                    # Extract items
                    for item in feedback_items:
                        span = item.find('span', class_='ng-star-inserted')
                        if span:
                            text = span.get_text(strip=True)
                            all_responses.append({
                                'page': page_num,
                                'text': text
                            })
                            print(f'  ✓ Extracted: {text[:60]}...')
                    
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    print(f'  ✗ Error on page {page_num} (attempt {retry_count}/{max_retries}): {str(e)[:100]}')
                    
                    if retry_count >= max_retries:
                        print(f'  ✗ Max retries reached for page {page_num}, skipping...')
                        success = True  # Move to next page
                    else:
                        await asyncio.sleep(2)  # Wait before retry
            
            # Save periodically
            if page_num % save_interval == 0:
                with open(INPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_responses, f, indent=2, ensure_ascii=False)
                print(f'  💾 Checkpoint: Saved {len(all_responses)} responses so far\n')
        
        await browser.close()
    
    # Final save
    with open(INPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)
    
    print(f'\n' + '='*60)
    print(f'✓ COMPLETE: Total {len(all_responses)} responses extracted')
    print(f'✓ Saved to: {INPUT_FILE}')
    print(f'='*60)

# Run it
if __name__ == '__main__':
    asyncio.run(scrape_consultations())
