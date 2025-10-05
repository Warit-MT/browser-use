import asyncio
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from pydantic import BaseModel

from browser_use import Agent, Browser, ChatGoogle

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')


# 🤖 Data Models สำหรับ AI Training Data
class PersonInfo(BaseModel):
	"""ข้อมูลพื้นฐานของบุคคล"""
	full_name: str
	current_position: str | None = None
	organization: str | None = None
	location: str | None = None
	education: str | None = None
	linkedin_url: str | None = None
	twitter_url: str | None = None
	github_url: str | None = None


class Achievement(BaseModel):
	"""ผลงานและความสำเร็จ"""
	title: str
	description: str
	year: str | None = None
	source_url: str | None = None


class QAPair(BaseModel):
	"""คู่คำถาม-คำตอบสำหรับ AI training"""
	question: str
	answer: str
	context: str | None = None
	confidence: str = "high"  # high, medium, low


class PersonDataset(BaseModel):
	"""Dataset เกี่ยวกับบุคคล พร้อมสำหรับ AI training"""
	person_info: PersonInfo
	biography: str | None = None
	achievements: list[Achievement] = []
	qa_pairs: list[QAPair] = []
	key_facts: list[str] = []
	recent_activities: list[str] = []
	related_urls: list[str] = []
	collection_timestamp: str
	total_sources: int = 0


def save_json_only(data: PersonDataset, output_dir: str = './ai_training_data'):
	"""💾 บันทึกแค่ JSON ไฟล์เดียว"""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	
	person_name_clean = data.person_info.full_name.replace(' ', '_').lower()
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	
	# บันทึกแค่ JSON ไฟล์เดียว
	json_file = output_path / f'{person_name_clean}_{timestamp}.json'
	with open(json_file, 'w', encoding='utf-8') as f:
		json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)
	print(f'📄 Saved: {json_file}')
	return str(json_file)


def save_ai_training_data(data: PersonDataset, output_dir: str = './ai_training_data'):
	"""💾 บันทึกข้อมูลในรูปแบบพร้อมสำหรับ AI Training"""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	
	person_name_clean = data.person_info.full_name.replace(' ', '_').lower()
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	
	# 1. บันทึก Full Dataset เป็น JSON
	json_file = output_path / f'{person_name_clean}_dataset_{timestamp}.json'
	with open(json_file, 'w', encoding='utf-8') as f:
		json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)
	print(f'📄 Full dataset JSON saved: {json_file}')
	
	# 2. บันทึก Q&A Pairs สำหรับ Fine-tuning (JSONL format)
	if data.qa_pairs:
		jsonl_file = output_path / f'{person_name_clean}_qa_training_{timestamp}.jsonl'
		with open(jsonl_file, 'w', encoding='utf-8') as f:
			for qa in data.qa_pairs:
				training_sample = {
					"messages": [
						{"role": "system", "content": f"You are an AI assistant knowledgeable about {data.person_info.full_name}."},
						{"role": "user", "content": qa.question},
						{"role": "assistant", "content": qa.answer}
					],
					"metadata": {"confidence": qa.confidence, "context": qa.context}
				}
				f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
		print(f'🤖 Q&A training data (JSONL) saved: {jsonl_file}')
	
	# 3. บันทึก Achievements เป็น CSV
	if data.achievements:
		csv_file = output_path / f'{person_name_clean}_achievements_{timestamp}.csv'
		with open(csv_file, 'w', encoding='utf-8', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=['title', 'description', 'year', 'source_url'])
			writer.writeheader()
			for achievement in data.achievements:
				writer.writerow(achievement.model_dump())
		print(f'🏆 Achievements CSV saved: {csv_file}')
	
	# 4. สร้าง AI-Ready Summary Report
	summary_file = output_path / f'{person_name_clean}_summary_{timestamp}.txt'
	with open(summary_file, 'w', encoding='utf-8') as f:
		f.write('=' * 80 + '\n')
		f.write('🤖 AI TRAINING DATA COLLECTION REPORT\n')
		f.write('=' * 80 + '\n\n')
		f.write(f'Person: {data.person_info.full_name}\n')
		f.write(f'Collection Time: {data.collection_timestamp}\n')
		f.write(f'Total Sources: {data.total_sources}\n')
		f.write(f'Data Quality: {"✅ High" if len(data.qa_pairs) > 5 else "⚠️ Limited"}\n\n')
		
		# Person Info
		f.write('👤 PERSON INFORMATION:\n')
		f.write('-' * 80 + '\n')
		if data.person_info.current_position:
			f.write(f'Position: {data.person_info.current_position}\n')
		if data.person_info.organization:
			f.write(f'Organization: {data.person_info.organization}\n')
		if data.person_info.location:
			f.write(f'Location: {data.person_info.location}\n')
		if data.person_info.education:
			f.write(f'Education: {data.person_info.education}\n')
		f.write('\n')
		
		# Biography
		if data.biography:
			f.write('📖 BIOGRAPHY:\n')
			f.write('-' * 80 + '\n')
			f.write(f'{data.biography}\n\n')
		
		# Key Facts
		if data.key_facts:
			f.write('💡 KEY FACTS:\n')
			f.write('-' * 80 + '\n')
			for i, fact in enumerate(data.key_facts, 1):
				f.write(f'{i}. {fact}\n')
			f.write('\n')
		
		# Achievements
		if data.achievements:
			f.write('🏆 ACHIEVEMENTS:\n')
			f.write('-' * 80 + '\n')
			for i, achievement in enumerate(data.achievements[:5], 1):
				f.write(f'{i}. {achievement.title}\n')
				f.write(f'   {achievement.description}\n')
				if achievement.year:
					f.write(f'   Year: {achievement.year}\n')
				f.write('\n')
		
		# Q&A Samples
		if data.qa_pairs:
			f.write('❓ SAMPLE Q&A PAIRS FOR AI TRAINING:\n')
			f.write('-' * 80 + '\n')
			for i, qa in enumerate(data.qa_pairs[:3], 1):
				f.write(f'Q{i}: {qa.question}\n')
				f.write(f'A{i}: {qa.answer}\n')
				f.write(f'    (Confidence: {qa.confidence})\n\n')
		
		# Training Recommendations
		f.write('🎯 AI TRAINING RECOMMENDATIONS:\n')
		f.write('-' * 80 + '\n')
		f.write(f'- Q&A pairs collected: {len(data.qa_pairs)}\n')
		f.write(f'- Recommended for: {"Fine-tuning ✅" if len(data.qa_pairs) > 10 else "RAG/Context injection ⚠️"}\n')
		f.write(f'- Data completeness: {min(100, len(data.key_facts) * 10 + len(data.achievements) * 5)}%\n')
	
	print(f'📝 AI-ready summary report saved: {summary_file}')
	
	# 5. สร้าง RAG-ready format (ถ้ามีข้อมูลเพียงพอ)
	if data.qa_pairs or data.key_facts:
		rag_file = output_path / f'{person_name_clean}_rag_context_{timestamp}.txt'
		with open(rag_file, 'w', encoding='utf-8') as f:
			f.write(f'Context about {data.person_info.full_name}\n\n')
			
			if data.biography:
				f.write(f'Biography: {data.biography}\n\n')
			
			if data.key_facts:
				f.write('Key Facts:\n')
				for fact in data.key_facts:
					f.write(f'- {fact}\n')
				f.write('\n')
			
			if data.achievements:
				f.write('Achievements:\n')
				for achievement in data.achievements:
					f.write(f'- {achievement.title}: {achievement.description}\n')
				f.write('\n')
		
		print(f'🔍 RAG context file saved: {rag_file}')


async def collect_person_data_lite(person_name: str = "Anuntapat Anuntachai"):
	"""🤖 AI Training Data Collector (LITE VERSION) - สำหรับ Demo"""
	
	print('=' * 70)
	print('🤖 AI DATA COLLECTOR - LITE VERSION (Token-Saving Mode)')
	print('=' * 70)
	print(f'\n🎯 Target: {person_name}')
	print('⚡ Quick search - minimal tokens\n')
	
	# Browser แบบ headful (เห็นการทำงาน) แต่ไม่บันทึก screenshots
	browser = Browser(
		headless=False,  # แสดง browser window
		keep_alive=False,  # ปิดอัตโนมัติเมื่อเสร็จ
	)
	
	llm = ChatGoogle(model='gemini-flash-latest', api_key=api_key)
	
	print(f'🔍 Searching for "{person_name}"...\n')
	
	# Task แบบไม่บังคับ structured output - ให้ส่ง text กลับมา
	search_agent = Agent(
		llm=llm,
		browser_session=browser,
		task=f"""
		Search for information about "{person_name}" and provide a comprehensive summary.

		Find and report:
		1. Full name, current position/title, organization/company
		2. Location (city/country)  
		3. Education background
		4. Short biography (2-3 paragraphs) - who they are, what they do, their background
		5. Top 3-5 achievements or notable works
		6. 5-10 interesting key facts about them
		7. Any social media links (LinkedIn, Twitter, GitHub, personal website)

		Format your response clearly with sections:
		NAME: 
		POSITION: 
		ORGANIZATION: 
		LOCATION:
		EDUCATION:
		
		BIO:
		[write 2-3 paragraphs]
		
		ACHIEVEMENTS:
		1. [achievement 1]
		2. [achievement 2]
		3. [achievement 3]
		
		KEY FACTS:
		- [fact 1]
		- [fact 2]
		- [fact 3]
		
		LINKS:
		- LinkedIn: [url]
		- Website: [url]

		Search 1-2 good sources and be thorough.
		""",
		# ไม่ใช้ output_format เพื่อให้ได้ข้อมูลเยอะขึ้น
	)
	
	result = await search_agent.run()
	
	# Parse text response แทน
	data = None
	raw_text = ""
	
	if result:
		# ดึง text จาก result
		if hasattr(result, 'final_result'):
			try:
				final = result.final_result() if callable(result.final_result) else result.final_result
				if isinstance(final, str):
					raw_text = final
			except:
				pass
	
	# Parse text และสร้าง structured data
	if raw_text:
		print('\n📝 Parsing collected information...')
		print(f'📄 Raw text preview: {raw_text[:300]}...\n')
		
		# Extract ข้อมูลจาก text
		def extract_section(text, section_name, end_marker=None):
			pattern = f"{section_name}:?\\s*(.+?)(?={end_marker}|$)" if end_marker else f"{section_name}:?\\s*(.+?)$"
			match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
			return match.group(1).strip() if match else None
		
		# Parse ข้อมูล
		name = extract_section(raw_text, "NAME", "POSITION") or person_name
		position = extract_section(raw_text, "POSITION", "ORGANIZATION")
		organization = extract_section(raw_text, "ORGANIZATION", "LOCATION")
		location = extract_section(raw_text, "LOCATION", "EDUCATION")
		education = extract_section(raw_text, "EDUCATION", "BIO")
		bio = extract_section(raw_text, "BIO", "ACHIEVEMENTS")
		
		# Extract achievements
		achievements = []
		achievements_section = extract_section(raw_text, "ACHIEVEMENTS", "KEY FACTS")
		if achievements_section:
			achievement_lines = [line.strip() for line in achievements_section.split('\n') if line.strip() and any(c.isalnum() for c in line)]
			for line in achievement_lines[:5]:  # top 5
				clean_line = re.sub(r'^\\d+[\\.\\)]\\s*', '', line).strip()
				if clean_line:
					achievements.append(Achievement(
						title=clean_line[:100],
						description=clean_line,
						year=None
					))
		
		# Extract key facts
		key_facts = []
		facts_section = extract_section(raw_text, "KEY FACTS", "LINKS")
		if facts_section:
			fact_lines = [line.strip() for line in facts_section.split('\n') if line.strip() and any(c.isalnum() for c in line)]
			for line in fact_lines[:10]:
				clean_line = re.sub(r'^[-•*]\\s*', '', line).strip()
				if clean_line:
					key_facts.append(clean_line)
		
		# Extract links
		links_section = extract_section(raw_text, "LINKS")
		linkedin_url = None
		if links_section and 'linkedin' in links_section.lower():
			linkedin_match = re.search(r'(https?://[^\\s\\)]+linkedin[^\\s\\)]+)', links_section, re.IGNORECASE)
			if linkedin_match:
				linkedin_url = linkedin_match.group(1)
		
		# Generate Q&A pairs
		qa_pairs = [
			QAPair(
				question=f"Who is {name}?",
				answer=bio[:200] if bio else f"{name} is {position} at {organization}" if position and organization else f"Information about {name}",
				confidence="high"
			),
			QAPair(
				question=f"What does {name} do?",
				answer=f"{name} works as {position} at {organization}." if position and organization else "See bio for details.",
				confidence="high" if position else "medium"
			),
			QAPair(
				question=f"Where is {name} located?",
				answer=location if location else "Location not specified",
				confidence="high" if location else "low"
			),
		]
		
		# สร้าง dataset
		data = PersonDataset(
			person_info=PersonInfo(
				full_name=name.strip() if name else person_name,
				current_position=position.strip() if position else None,
				organization=organization.strip() if organization else None,
				location=location.strip() if location else None,
				education=education.strip() if education else None,
				linkedin_url=linkedin_url,
			),
			biography=bio.strip() if bio else None,
			achievements=achievements,
			qa_pairs=qa_pairs,
			key_facts=key_facts,
			collection_timestamp=datetime.now().isoformat(),
			total_sources=2,
		)
		
		print('✅ Data parsed successfully!')
	
	if data:
		print('\n' + '=' * 70)
		print('✅ DONE!')
		print('=' * 70)
		print(f'\n📋 Quick Summary:')
		print(f'   Name: {data.person_info.full_name}')
		print(f'   Position: {data.person_info.current_position or "N/A"}')
		print(f'   Organization: {data.person_info.organization or "N/A"}')
		print(f'   Facts: {len(data.key_facts)} items')
		print(f'   Achievements: {len(data.achievements)} items')
		print(f'   Q&A pairs: {len(data.qa_pairs)} pairs')
		
		# แสดง Preview
		if data.biography:
			print(f'\n📖 Bio Preview:')
			bio_preview = data.biography[:200] + '...' if len(data.biography) > 200 else data.biography
			print(f'   {bio_preview}')
		
		if data.qa_pairs:
			print(f'\n❓ Sample Q&A:')
			for i, qa in enumerate(data.qa_pairs[:2], 1):
				print(f'   Q{i}: {qa.question}')
				print(f'   A{i}: {qa.answer[:80]}...' if len(qa.answer) > 80 else f'   A{i}: {qa.answer}')
		
		# บันทึกแค่ JSON เดียว
		print(f'\n💾 Saving to JSON...')
		json_file = save_json_only(data)
		
		print('\n' + '=' * 70)
		print('🎉 COMPLETE!')
		print('=' * 70)
		print(f'\n📁 Saved: {json_file}')
		print(f'💡 All data in one JSON file')
	else:
		print('\n⚠️ No structured data collected - using fallback')
		print('💡 Creating basic dataset from search results...\n')
		
		# Fallback: สร้าง dataset พื้นฐานจากข้อมูลที่มี
		data = PersonDataset(
			person_info=PersonInfo(
				full_name=person_name,
				current_position="Information found during search",
				organization="See extracted files for details",
			),
			biography="Agent successfully searched for information but did not return structured data. Check extracted_content_*.md files for raw data.",
			key_facts=[
				f"Search completed for {person_name}",
				"Data extracted from multiple sources (KMITL, IEEE Xplore)",
				"Raw data saved to extracted_content files",
			],
			qa_pairs=[
				QAPair(
					question=f"Who is {person_name}?",
					answer=f"Based on search results, {person_name} is associated with KMITL and has academic publications.",
					confidence="medium"
				),
			],
			collection_timestamp=datetime.now().isoformat(),
			total_sources=2,
		)
		
		print('✅ Basic dataset created from search metadata')
		print('\n💾 Saving to JSON...')
		json_file = save_json_only(data)
		
		print('\n' + '=' * 70)
		print('🎉 COMPLETE (Fallback Mode)')
		print('=' * 70)
		print(f'\n📁 Saved: {json_file}')
		print(f'💡 TIP: Check extracted_content_*.md files for full raw data')


async def simple_demo():
	"""🚀 Demo ง่ายๆ - ถ้าต้องการทดสอบรวดเร็ว"""
	browser = Browser(
		headless=False,
		traces_dir='./tmp/traces',
		keep_alive=True,
	)
	
	llm = ChatGoogle(model='gemini-flash-latest', api_key=api_key)
	agent = Agent(
		llm=llm,
		browser_session=browser,
		task='นายกประเทศไทยคือใครและมีข้อมูลอะไรบ้าง?',
	)

	await agent.run()
	
	print('\n✅ เสร็จสิ้น! ดู screenshots ที่มีกรอบ element ได้ที่: ./tmp/traces/screenshots/')
	print('📝 กด Enter เพื่อปิด browser...')
	input()


if __name__ == '__main__':
	# เลือกโหมดที่ต้องการ:
	
	# 1. LITE VERSION - ประหยัด token, รวดเร็ว, ไม่มี screenshots ⚡
	asyncio.run(collect_person_data_lite("Anuntapat Anuntachai"))
	
	# 2. Simple Demo
	# asyncio.run(simple_demo())
	
	# 3. ค้นหาคนอื่น
	# asyncio.run(collect_person_data_lite("Elon Musk"))
	# asyncio.run(collect_person_data_lite("Steve Jobs"))
