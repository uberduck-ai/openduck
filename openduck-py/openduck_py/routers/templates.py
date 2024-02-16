from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
import copy
import os

from pydantic import BaseModel
from fastapi import HTTPException, APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import NoResultFound, IntegrityError
from openai import AsyncAzureOpenAI
import jinja2
from jinja2.meta import find_undeclared_variables

from openduck_py.models import (
    DBTemplatePrompt,
    DBTemplateDeployment,
    DBUser,
)
from openduck_py.db import get_db_async

from openduck_py.auth import get_user_sqlalchemy
from openduck_py.utils.utils import make_url_name

client = AsyncAzureOpenAI(
    azure_endpoint="https://uberduck-azure-openai.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
)


templates_router = APIRouter(prefix="/templates")

ModelLiteral = Literal["gpt-35-turbo-deployment", "gpt-4-deployment"]
DEFAULT_MODEL: ModelLiteral = "gpt-35-turbo-deployment"


class TemplateRequest(BaseModel):
    display_name: str
    prompt: Optional[str] = ""
    variables: Optional[List[str]] = None
    values: Optional[List[Dict[str, str]]] = None
    model: Optional[ModelLiteral] = None


class EditRequest(BaseModel):
    """
    For the PATCH request, each payload field is optional because the user might only update a
    subset of fields.
    """

    display_name: Optional[str] = None
    prompt: Optional[str] = None
    variables: Optional[List[str]] = None
    values: Optional[List[Dict[str, str]]] = None
    model: Optional[ModelLiteral] = None


class DeployRequest(BaseModel):
    display_name: str


class TemplateResponse(BaseModel):
    uuid: str
    display_name: str
    url_name: str
    created_at: int
    variables: List[str]
    values: List[Dict[str, str]]
    prompt: Dict[str, Any]
    completion: Optional[str] = None
    model: Optional[ModelLiteral] = None


class GenerationRequest(BaseModel):
    variables: Dict[str, str]


class GenerationResponse(BaseModel):
    """
    OpenAI Chat Completions response format:
    https://platform.openai.com/docs/guides/text-generation/chat-completions-api
    """

    choices: List
    created: int
    id: str
    model: str
    object: str
    usage: Any


class StatusResponse(BaseModel):
    status: str


class GetTemplatesResponse(BaseModel):
    templates: List[TemplateResponse]


def create_template_response(
    template: Union[DBTemplatePrompt, DBTemplateDeployment]
) -> TemplateResponse:
    return TemplateResponse(
        uuid=template.uuid,
        display_name=template.display_name,
        url_name=template.url_name,
        created_at=int(template.created_at.timestamp()),
        variables=template.meta_json["variables"],
        values=template.meta_json["values"],
        prompt=template.prompt,
        completion=template.meta_json.get("completion", [""])[0],
        model=template.model,
    )


def select_prompt(db: AsyncSession, user: DBUser, uuid: str) -> DBTemplatePrompt:
    return select(DBTemplatePrompt).filter(
        DBTemplatePrompt.uuid == uuid,
        DBTemplatePrompt.user_id == user.id,
        DBTemplatePrompt.deleted_at == None,
    )


def select_deployment(
    db: AsyncSession, user: DBUser, url_name: str
) -> DBTemplateDeployment:
    return select(DBTemplateDeployment).filter(
        DBTemplateDeployment.url_name == url_name,
        DBTemplateDeployment.user_id == user.id,
        DBTemplateDeployment.deleted_at == None,
    )


# TODO (Matthew): do this in a more normal way
async def execute_or_error(db, statement):
    try:
        result = await db.execute(statement)
        db_template = result.scalars().one()
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Template not found")
    return db_template


def chat_completion_format(prompt: str) -> Dict[str, List[Dict[str, str]]]:
    return {"messages": [{"role": "user", "content": prompt}]}


def check_prompt_variables(prompt: str, variables: List[str]):
    env = jinja2.Environment()
    parsed_content = env.parse(prompt)
    template_variables = list(find_undeclared_variables(parsed_content))
    check_variables(template_variables, variables)


@templates_router.post("/prompts")
async def create_template(
    template_request: TemplateRequest,
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> TemplateResponse:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(
    #     user.email,
    #     "[Templates] Created template prompt",
    # )
    url_name = make_url_name(template_request.display_name)

    check_prompt_variables(template_request.prompt, template_request.variables)

    try:
        new_template = DBTemplatePrompt(
            user_id=user.id,
            url_name=url_name,
            display_name=template_request.display_name,
            prompt=chat_completion_format(template_request.prompt),
            meta_json={
                "variables": template_request.variables,
                "values": template_request.values,
            },
            model=template_request.model,
            deleted_at=None,
        )
        db.add(new_template)
        await db.commit()
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Prompt must have a unique name")

    return create_template_response(new_template)


@templates_router.get("/prompts")
async def get_prompts(
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> GetTemplatesResponse:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(user.email, "[Templates] Get prompt request")
    statement = (
        select(DBTemplatePrompt)
        .filter(
            DBTemplatePrompt.user_id == user.id,
            DBTemplatePrompt.deleted_at == None,
        )
        .order_by(DBTemplatePrompt.created_at.desc())
    )
    db_templates = await db.execute(statement)
    prompts = db_templates.scalars().all()
    prompts_list = []
    for prompt in prompts:
        prompts_list.append(create_template_response(prompt))
    return GetTemplatesResponse(templates=prompts_list)


@templates_router.get("/prompts/{uuid}")
async def get_prompt(
    uuid: str,
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> GetTemplatesResponse:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(
    #     user.email,
    #     "[Templates] Get prompt request",
    #     dict(promptUUID=uuid),
    # )
    statement = select_prompt(db, user, uuid).order_by(
        DBTemplatePrompt.created_at.desc()
    )
    db_template = await execute_or_error(db, statement)
    prompt = create_template_response(db_template)
    return GetTemplatesResponse(templates=[prompt])


@templates_router.get("/deployments")
async def get_deployments(
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> GetTemplatesResponse:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(
    #     user.email,
    #     "[Templates] Get deployment request",
    # )
    statement = (
        select(DBTemplateDeployment)
        .filter(
            DBTemplateDeployment.user_id == user.id,
            DBTemplateDeployment.deleted_at == None,
        )
        .order_by(DBTemplateDeployment.created_at.desc())
    )
    db_templates = await db.execute(statement)
    deployments = db_templates.scalars().all()
    deployments_list = []
    for deployment in deployments:
        deployments_list.append(create_template_response(deployment))
    return GetTemplatesResponse(templates=deployments_list)


@templates_router.get("/deployments/{url_name}")
async def get_deployment(
    url_name: str,
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> GetTemplatesResponse:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(
    #     user.email,
    #     "[Templates] Get deployment request",
    #     dict(deploymentUrlName=url_name),
    # )
    statement = select_deployment(db, user, url_name).order_by(
        DBTemplateDeployment.created_at.desc()
    )
    db_template = await execute_or_error(db, statement)
    deployment = create_template_response(db_template)
    return GetTemplatesResponse(templates=[deployment])


@templates_router.patch("/prompts/{uuid}")
async def edit_prompt(
    uuid: str,
    edit_request: EditRequest,
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> TemplateResponse:
    # await track_async(user.email, "[Templates] Edited template prompt")
    statement = select_prompt(db, user, uuid)
    db_template = await execute_or_error(db, statement)
    if edit_request.display_name is not None:
        db_template.display_name = edit_request.display_name
        url_name = make_url_name(edit_request.display_name)
        db_template.url_name = url_name
    if edit_request.prompt is not None:
        db_template.prompt = chat_completion_format(edit_request.prompt)
    if edit_request.variables is not None:
        check_prompt_variables(db_template.prompt, edit_request.variables)
        db_template.meta_json["variables"] = edit_request.variables
    if edit_request.values is not None:
        db_template.meta_json["values"] = edit_request.values

    if edit_request.model is not None:
        db_template.model = edit_request.model

    await db.commit()
    return create_template_response(db_template)


@templates_router.delete("/prompts/{uuid}")
async def delete_prompt(
    uuid: str,
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> StatusResponse:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(
    #     user.email,
    #     "[Templates] Deleted template prompt",
    #     dict(promptUUID=uuid),
    # )
    statement = select_prompt(db, user, uuid)
    db_template = await execute_or_error(db, statement)
    db_template.deleted_at = datetime.utcnow()
    await db.commit()
    return StatusResponse(status="OK")


@templates_router.delete("/deployments/{url_name}")
async def delete_deployment(
    url_name: str,
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> StatusResponse:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(
    #     user.email,
    #     "[Templates] Deleted template deployment",
    #     dict(deploymentUrlName=url_name),
    # )
    statement = select_deployment(db, user, url_name)
    db_template = await execute_or_error(db, statement)
    db_template.deleted_at = datetime.utcnow()
    await db.commit()
    return StatusResponse(status="OK")


async def generate(
    template: Dict[str, List[Dict[str, str]]],
    variables: Dict[str, Any],
    model: ModelLiteral,
) -> GenerationResponse:
    print(f"Generating completion: {template}, {variables}, {model}")
    # Substitute variables for every chat in the template
    prompt = copy.deepcopy(template["messages"])
    for i, chat in enumerate(prompt):
        text = chat["content"]
        jinja_template = jinja2.Template(text)
        prompt[i]["content"] = jinja_template.render(variables)

    response = await client.chat.completions.create(
        model=model or DEFAULT_MODEL, messages=prompt, temperature=0.3
    )
    return response


def check_variables(
    template_variable_names: List[str], input_variable_names: List[str]
):
    template_set = set(template_variable_names)
    input_set = set(input_variable_names)
    missing_vars = template_set - input_set
    extra_vars = input_set - template_set
    if missing_vars:
        raise HTTPException(
            status_code=400, detail=f"Missing the following variables: {missing_vars}"
        )
    if extra_vars:
        raise HTTPException(
            status_code=400, detail=f"Specified variables not in template: {extra_vars}"
        )

    if len(template_variable_names) != len(set(template_variable_names)):
        raise HTTPException(
            status_code=400, detail="Template variables contain duplicates"
        )
    if len(input_variable_names) != len(set(input_variable_names)):
        raise HTTPException(
            status_code=400, detail="Input variables contain duplicates"
        )


@templates_router.post("/prompts/{uuid}/generate")
async def prompt_generate(
    uuid: str,
    generation_request: GenerationRequest,
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> Optional[GenerationResponse]:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(
    #     user.email,
    #     "[Templates] Generated completion from prompt",
    #     dict(promptUUID=uuid),
    # )
    statement = select_prompt(db, user, uuid)
    db_template = await execute_or_error(db, statement)
    check_variables(
        db_template.meta_json["variables"], list(generation_request.variables.keys())
    )

    response = await generate(
        db_template.prompt, generation_request.variables, db_template.model
    )

    # Set the DBTemplatePrompt's completion field to the LLM's completion string
    completion = response.choices[0].message.content
    db_template.meta_json["completion"] = [completion]
    await db.commit()

    # TODO(Matthew): Add model use to DBUserUsage in prompt_generate() and deployment_generate()
    return response


@templates_router.post("/deployments/{url_name}/generate")
async def deployment_generate(
    url_name: str,
    generation_request: GenerationRequest,
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> Optional[GenerationResponse]:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(
    #     user.email,
    #     "[Templates] Generated completion from deployment",
    #     dict(deploymentUrlName=url_name),
    # )
    statement = select_deployment(db, user, url_name)
    db_template = await execute_or_error(db, statement)
    check_variables(
        db_template.meta_json["variables"],
        list(generation_request.variables.keys()),
    )

    response = await generate(
        db_template.prompt, generation_request.variables, db_template.model
    )

    model = db_template.model or DEFAULT_MODEL
    return response


@templates_router.post("/prompts/{prompt_uuid}/deploy")
async def deploy_prompt(
    prompt_uuid: str,
    deploy_request: DeployRequest,
    user: DBUser = Depends(get_user_sqlalchemy),
    db: AsyncSession = Depends(get_db_async),
) -> TemplateResponse:
    # await aio_rate_limit(user, rate_limit_type="templates")
    # await track_async(
    #     user.email,
    #     "[Templates] Deployed prompt",
    #     dict(promptUUID=prompt_uuid),
    # )
    statement = select_prompt(db, user, prompt_uuid)
    db_prompt = await execute_or_error(db, statement)
    deploy_url_name = make_url_name(deploy_request.display_name)
    try:
        # Modify the existing deployment if it already exists
        statement = select_deployment(db, user, deploy_url_name)
        db_deployment = (await db.execute(statement)).scalars().one()
        db_deployment.prompt = db_prompt.prompt
        db_deployment.meta_json = db_prompt.meta_json
        db_deployment.model = db_prompt.model
    except NoResultFound:
        db_deployment = DBTemplateDeployment(
            user_id=user.id,
            url_name=deploy_url_name,
            display_name=deploy_request.display_name,
            prompt=db_prompt.prompt,
            meta_json=db_prompt.meta_json,
            model=db_prompt.model,
        )
    db.add(db_deployment)
    await db.commit()
    return create_template_response(db_deployment)
